"""Kinara live bone stream constants and protocol definitions."""
import struct

PORT = 49168
LIMIT = 1024 * 1024
SNAPSHOT = struct.Struct('<QH')
ACTOR_FRAME = struct.Struct('<II')
SELECTION_COUNT = struct.Struct('<H')
ACTOR_ID_BYTES = 32
MAX_ACTORS = 128
MAX_BAKE_FRAMES = 36000
MAX_BAD_FRAMES = 60          # consecutive snapshots with an unusable pose before disconnecting
BAKE_TIMEOUT = 15.0          # seconds without bake packets before a pending bake is abandoned
COLLECTION = 'Kinara Stream'
OWNER = 'kinara_bone_stream'
SKELETON = 'kinara_stream_skeleton'
ACTOR = 'kinara_stream_actor'
BONE_INDEX = 'kinara_stream_bone_index'
REGISTRY_PREFIX = 'stream_rig_'

# Errors that mean "this connection or job is bad", not "the add-on has a bug".
RECOVERABLE = (OSError, ValueError, TypeError, KeyError, RuntimeError, ReferenceError, struct.error)


class PoseError(ValueError):
    """A single pose frame is unusable. The stream itself is still fine."""
