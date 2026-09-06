# Per-episode prompt generation for the videogen pipeline.
# Template mirrors renderscale prompt_builder.py: background sentence,
# foreground objects on the surface, robot instruction. Deterministic per seed.
import random

WALLS = [
    "light gray walls",
    "warm white walls with subtle texture",
    "pale blue-gray walls",
    "beige plaster walls",
    "soft sage-green walls",
    "off-white walls with a large window casting soft light",
    "light taupe walls with white trim",
]

FLOORS = [
    "a light yellow wood floor",
    "a dark walnut plank floor",
    "a polished gray concrete floor",
    "a white large-tile floor",
    "a honey-colored oak parquet floor",
    "a light bamboo floor",
    "a terracotta tile floor",
    "a pale gray laminate floor",
]

TABLES = [
    "the white table",
    "a warm walnut butcher-block table",
    "a speckled gray stone table",
    "a matte black table",
    "a sun-bleached pine table",
    "a polished white marble table with subtle gray veins",
    "a brushed stainless steel table",
    "a varnished oak farmhouse table",
]

OBJECTS = [
    ("a can of potato chips with a glossy red label and silver rim", "potato chips can"),
    ("a tall emerald-green anodized aluminum soda can with a printed paper label", "green soda can"),
    ("a bright blue aluminum energy drink can with a matte finish", "blue can"),
    ("an orange cylindrical snack container with a white plastic lid", "orange container"),
    ("a silver coffee tin with a black embossed logo", "coffee tin"),
    ("a yellow lemonade can with condensation droplets on its surface", "yellow can"),
    ("a white ceramic-look canister with thin blue stripes", "striped canister"),
]

CONTAINERS = [
    ("a black storage basket with a matte woven-plastic texture and visible weave pattern", "black woven basket"),
    ("a natural wicker basket with a warm honey tone", "wicker basket"),
    ("a navy-blue fabric storage bin with white stitched edges", "blue fabric bin"),
    ("a light gray felt basket with rounded corners", "gray felt basket"),
    ("a rustic wooden crate with visible grain and slats", "wooden crate"),
    ("a white perforated metal basket", "white metal basket"),
]


def build_prompt(seed: int) -> str:
    rng = random.Random(int(seed))
    wall = rng.choice(WALLS)
    floor = rng.choice(FLOORS)
    table = rng.choice(TABLES)
    obj_long, obj_short = rng.choice(OBJECTS)
    cont_long, cont_short = rng.choice(CONTAINERS)
    return (
        f"This video shows in the background, a bright room with {wall} and {floor}. "
        f"In the foreground, on {table}, there is {obj_long} and {cont_long}. "
        f"The robot is instructed to pick up the {obj_short} and place it into the {cont_short}."
    )


if __name__ == "__main__":
    for s in range(4):
        print(f"[{s}] {build_prompt(s)}\n")
