"""Pure CPU stubs: no torch/Kit/PhysX imports or native initialization."""
from test.deployment.source_paths import ACTOR_ROOT
from types import SimpleNamespace as NS
import ast
import json
from pathlib import Path
import unittest

try:
    from .native_physics_readback import MAX_VALUES, _plain, drive_readback, physics_readback
except ImportError:
    from native_physics_readback import MAX_VALUES, _plain, drive_readback, physics_readback


class View:
    def __init__(self): self.calls=[]
    def get_dof_stiffnesses(self): self.calls.append('stiffness'); return [[.5,.5,2.],[.5,.5,2.]]
    def get_dof_dampings(self): self.calls.append('damping'); return [[.1,.1,.1],[.1,.1,.1]]
    def get_dof_max_forces(self): self.calls.append('effort'); return [[1.,1.,1.],[1.,1.,1.]]
    def get_dof_max_velocities(self): self.calls.append('vel'); return [[5.,5.,5.],[5.,5.,5.]]
    def get_dof_position_targets(self): self.calls.append('position'); return [[.8,.9,0],[.8,.9,0]]
    def get_dof_velocity_targets(self): self.calls.append('velocity'); return [[0,0,0],[0,0,0]]
    def get_dof_actuation_forces(self): self.calls.append('actuation'); return [[0,0,0],[0,0,0]]
    def get_dof_projected_joint_forces(self): self.calls.append('projected'); return [[.2,.1,.3],[.1,.2,.3]]
    def get_masses(self): self.calls.append('mass'); return [[.1],[.1]]
    def get_inertias(self): return [[[1,0,0,0,1,0,0,0,1]],[[1,0,0,0,1,0,0,0,1]]]
    def get_material_properties(self): return [[[.5,.5,0]],[[.6,.4,0]]]
    def get_contact_offsets(self): return [[.002],[.002]]
    def get_rest_offsets(self): return [[0],[0]]


def environment():
    view=View()
    actuator=NS(joint_names=['a','b','c'],joint_indices=slice(None),is_implicit_model=True,
        stiffness=[[.5,.5,2.]]*2,damping=[[.1,.1,.1]]*2,effort_limit_sim=[[1.,1.,1.]]*2,
        velocity_limit_sim=[[5.,5.,5.]]*2,armature=[[.01]*3]*2,friction=[[0]*3]*2)
    robot=NS(root_view=view,joint_names=['a','b','c'],_has_implicit_actuators=True,
        data=NS(joint_pos_target=[[999,999,999]]*2),_joint_pos_target_sim=[[.8,.9,0]]*2,
        actuators={'all':actuator})
    tasks={key:NS(root_view=View()) for key in ('front','hold','tray')}
    sim=NS(_robot=robot,dof_names=['c','a','b'],dof_ids=[2,0,1],_task=tasks)
    return NS(num_envs=2,simulator=sim,config=NS(task=NS(front_object='front',hold_object='hold',tray_object='tray')),
        _right_hand_dof_idx=[1,2]),view


class ReadbackTests(unittest.TestCase):
    def test_exact_mapping_and_both_lane_properties(self):
        env,v=environment();result=physics_readback(env)
        self.assertTrue(result['joint_mapping']['mapping_names_match'])
        self.assertEqual(result['joint_mapping']['simulator_native_joint_indices'],[2,0,1])
        self.assertEqual(result['robot_backend']['get_dof_stiffnesses']['value'],[[.5,.5,2.]]*2)
        self.assertEqual(result['has_implicit_actuators'],{'status':'ok','value':True})
        self.assertTrue(result['read_only'])
        self.assertTrue(all(len(result['tasks_backend'][key]['get_masses']['value'])==2 for key in result['tasks_backend']))

    def test_backend_targets_not_user_buffer(self):
        env,v=environment();r=drive_readback(env)
        self.assertNotEqual(r['backend']['get_dof_position_targets']['value'],r['user_position_target']['value'])
        self.assertEqual(r['backend']['get_dof_position_targets']['value'],r['staging_position_target']['value'])
        self.assertIn('not certified total implicit',r['force_semantics']['get_dof_actuation_forces'])
        self.assertEqual(set(v.calls),{'position','velocity','actuation','projected'})

    def test_missing_getter_not_reported_as_zero(self):
        env,v=environment();r=physics_readback(env)
        self.assertEqual(r['robot_backend']['get_dof_armatures']['status'],'unsupported')
        self.assertNotIn('value',r['robot_backend']['get_dof_armatures'])
        self.assertEqual(r['usd']['status'],'unsupported')
        json.dumps(r,allow_nan=False)

    def test_nonfinite_and_errors_are_sanitized(self):
        env,v=environment()
        v.get_dof_position_targets=lambda:[[float('nan')]]
        def bad():raise RuntimeError('Bearer secret-never-print')
        v.get_dof_velocity_targets=bad
        r=drive_readback(env)
        self.assertEqual(r['backend']['get_dof_position_targets']['status'],'unavailable')
        self.assertEqual(r['backend']['get_dof_velocity_targets']['error_type'],'RuntimeError')
        self.assertNotIn('secret',json.dumps(r))

    def test_oversized_array_rejected_before_copy(self):
        class Large:
            shape=(MAX_VALUES+1,)
            def numpy(self):raise AssertionError('must not copy oversized data')
        with self.assertRaises(ValueError):_plain(Large())

    def test_b3_and_bad_joint_map_rejected(self):
        env,v=environment();env.num_envs=3
        with self.assertRaises(ValueError):physics_readback(env)
        env.num_envs=2;env.simulator.dof_ids=[1,2,7]
        with self.assertRaises(ValueError):physics_readback(env)

    def test_missing_usd_paths_are_explicit_for_both_lanes(self):
        class Prim:
            def IsValid(self):return False
        class Stage:
            def __init__(self):self.paths=[]
            def GetPrimAtPath(self,path):self.paths.append(path);return Prim()
        env,v=environment();stage=Stage();env.simulator.sim=NS(stage=stage)
        r=physics_readback(env)['usd']
        self.assertEqual(len(r['rows']),10)
        self.assertEqual({row['lane'] for row in r['rows']},{0,1})
        self.assertTrue(all(row['status']=='unsupported' for row in r['rows']))

    def test_no_simulation_or_setter_calls_in_module(self):
        tree=ast.parse((ACTOR_ROOT / 'native_physics_readback.py').read_text())
        names=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(any(n.startswith(('set_','write_')) or n in ('step','forward','update','reset') for n in names))


if __name__=='__main__':unittest.main()
