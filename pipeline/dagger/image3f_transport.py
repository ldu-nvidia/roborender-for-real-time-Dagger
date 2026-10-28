"""Opt-in same-process COMPLETE-reply recovery; never re-render on reconnect.

Channels are bound to their original TCP connection. A different connection
may recover only the latest, identical, completed reply. Successful recovery
binds the channel to that connection for its next strictly increasing request.
In-flight, failed, unseen, evicted and different-process work fails closed.
"""
from collections import OrderedDict
import hashlib
import threading
import uuid

SCHEMA = 'roborender.image3f.cached-reply.v1'


class ReplyRecoveryError(RuntimeError):
    pass


def identifier(value):
    if not isinstance(value, str) or len(value) != 32 or any(c not in '0123456789abcdef' for c in value):
        raise ReplyRecoveryError('invalid renderer transport identity')
    return value


class CompleteReplyCache:
    def __init__(self, *, maximum_replies=64, maximum_channels=4096):
        if type(maximum_replies) is not int or not 1 <= maximum_replies <= 128:
            raise ValueError('bounded renderer reply capacity required')
        if type(maximum_channels) is not int or not 1 <= maximum_channels <= 4096:
            raise ValueError('bounded renderer channel capacity required')
        self.instance = uuid.uuid4().hex
        self.maximum_replies, self.maximum_channels = maximum_replies, maximum_channels
        self.channels = {}
        self.replies = OrderedDict()
        self.lock = threading.Lock()

    def open_channel(self, channel, connection):
        identifier(channel); identifier(connection)
        with self.lock:
            existing = self.channels.get(channel)
            if existing is not None and existing['sequence'] != -1:
                raise ReplyRecoveryError('a used renderer channel cannot be reopened')
            if existing is None and len(self.channels) >= self.maximum_channels:
                raise ReplyRecoveryError('renderer channel capacity reached; no eviction of admission history')
            self.channels[channel] = dict(connection=connection, sequence=-1, state='ready', digest=None)
            return dict(schema=SCHEMA, server_instance=self.instance, channel=channel)

    def execute(self, binding, payload, connection, produce):
        if (not isinstance(binding, dict)
                or set(binding) != {'schema','server_instance','channel','sequence','payload_sha256'}
                or binding['schema'] != SCHEMA or binding['server_instance'] != self.instance
                or type(binding['sequence']) is not int or binding['sequence'] < 0
                or not isinstance(payload, bytes) or not 0 < len(payload) <= 64 << 20
                or hashlib.sha256(payload).hexdigest() != binding['payload_sha256']):
            raise ReplyRecoveryError('renderer process/request binding differs')
        channel=identifier(binding['channel']); identifier(connection)
        sequence=binding['sequence']; digest=binding['payload_sha256']; key=(channel,sequence)
        with self.lock:
            row=self.channels.get(channel)
            if row is None:
                raise ReplyRecoveryError('renderer request channel was never admitted')
            if sequence == row['sequence']:
                if digest != row['digest']:
                    raise ReplyRecoveryError('renderer request content changed on retry')
                if row['state'] != 'complete' or key not in self.replies:
                    raise ReplyRecoveryError('renderer reply is in-flight, failed or evicted; recompute forbidden')
                row['connection']=connection
                self.replies.move_to_end(key)
                return self.replies[key]
            if (sequence != row['sequence'] + 1 or connection != row['connection']
                    or row['state'] not in {'ready','complete'}):
                raise ReplyRecoveryError('reconnected renderer request has no COMPLETE reply; recompute forbidden')
            row.update(sequence=sequence,digest=digest,state='inflight')
        try:
            response=produce()
            if not isinstance(response,dict) or response.get('ok') is not True:
                raise ReplyRecoveryError('renderer did not complete successfully')
            complete={**response,'transport_receipt':{**binding,'state':'complete'}}
        except BaseException:
            with self.lock:
                row['state']='failed'
            raise
        with self.lock:
            # Publish the full immutable RGB reply before any socket send.
            self.replies[key]=complete
            row['state']='complete'
            while len(self.replies)>self.maximum_replies:
                self.replies.popitem(last=False)
            return complete
