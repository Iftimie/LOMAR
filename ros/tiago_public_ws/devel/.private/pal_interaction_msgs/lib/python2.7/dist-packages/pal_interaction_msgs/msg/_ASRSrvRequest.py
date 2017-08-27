# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from pal_interaction_msgs/ASRSrvRequest.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import pal_interaction_msgs.msg

class ASRSrvRequest(genpy.Message):
  _md5sum = "18340721947db95a89c5d69f8dcbb2cc"
  _type = "pal_interaction_msgs/ASRSrvRequest"
  _has_header = False #flag to mark the presence of a Header object
  _full_text = """# Request messages for the recognizer service.
# It is possible to request and activate task, 
# a grammar management task and language change or just
# request the current status.

# Type of request list
int8 ACTIVATION = 1
int8 GRAMMAR = 2
int8 LANGUAGE = 3
int8 STATUS = 4
int8 KWSPOTTING = 5

# Message variables
# list of requests types (several requests can be send in one single message)
int8[] requests

# Information related to each possible request
# except for status that does not need any additional info.
ASRActivation activation
ASRLangModelMngmt model
ASRLanguage lang

================================================================================
MSG: pal_interaction_msgs/ASRActivation
# Message that can be used to send activation commands to the speech recognizer.
# It is possible to activate/deactivate or pause/resume the recognizer with these commands.
# action list
int8 ACTIVATE = 1
int8 DEACTIVATE = 2
int8 PAUSE = 3
int8 RESUME = 4
int8 CALIBRATE = 5

# Message variables
int8 action


================================================================================
MSG: pal_interaction_msgs/ASRLangModelMngmt
# This message is to be used in the ASR service to manage the grammars
# makes possible to enable/disable, load/unload grammars.


# Types of action
int8 ENABLE = 1
int8 DISABLE = 2
int8 LOAD = 3
int8 UNLOAD = 4

# Message variables
# Type of action requested
int8 action

# Name of the grammar to actuate on.
string modelName

================================================================================
MSG: pal_interaction_msgs/ASRLanguage
# This message is to indicate the language
# that has to be set in the speech recognizer
string language
"""
  # Pseudo-constants
  ACTIVATION = 1
  GRAMMAR = 2
  LANGUAGE = 3
  STATUS = 4
  KWSPOTTING = 5

  __slots__ = ['requests','activation','model','lang']
  _slot_types = ['int8[]','pal_interaction_msgs/ASRActivation','pal_interaction_msgs/ASRLangModelMngmt','pal_interaction_msgs/ASRLanguage']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       requests,activation,model,lang

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(ASRSrvRequest, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.requests is None:
        self.requests = []
      if self.activation is None:
        self.activation = pal_interaction_msgs.msg.ASRActivation()
      if self.model is None:
        self.model = pal_interaction_msgs.msg.ASRLangModelMngmt()
      if self.lang is None:
        self.lang = pal_interaction_msgs.msg.ASRLanguage()
    else:
      self.requests = []
      self.activation = pal_interaction_msgs.msg.ASRActivation()
      self.model = pal_interaction_msgs.msg.ASRLangModelMngmt()
      self.lang = pal_interaction_msgs.msg.ASRLanguage()

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      length = len(self.requests)
      buff.write(_struct_I.pack(length))
      pattern = '<%sb'%length
      buff.write(struct.pack(pattern, *self.requests))
      _x = self
      buff.write(_struct_2b.pack(_x.activation.action, _x.model.action))
      _x = self.model.modelName
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      if python3:
        buff.write(struct.pack('<I%sB'%length, length, *_x))
      else:
        buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.lang.language
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      if python3:
        buff.write(struct.pack('<I%sB'%length, length, *_x))
      else:
        buff.write(struct.pack('<I%ss'%length, length, _x))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      if self.activation is None:
        self.activation = pal_interaction_msgs.msg.ASRActivation()
      if self.model is None:
        self.model = pal_interaction_msgs.msg.ASRLangModelMngmt()
      if self.lang is None:
        self.lang = pal_interaction_msgs.msg.ASRLanguage()
      end = 0
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sb'%length
      start = end
      end += struct.calcsize(pattern)
      self.requests = struct.unpack(pattern, str[start:end])
      _x = self
      start = end
      end += 2
      (_x.activation.action, _x.model.action,) = _struct_2b.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.model.modelName = str[start:end].decode('utf-8')
      else:
        self.model.modelName = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.lang.language = str[start:end].decode('utf-8')
      else:
        self.lang.language = str[start:end]
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      length = len(self.requests)
      buff.write(_struct_I.pack(length))
      pattern = '<%sb'%length
      buff.write(self.requests.tostring())
      _x = self
      buff.write(_struct_2b.pack(_x.activation.action, _x.model.action))
      _x = self.model.modelName
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      if python3:
        buff.write(struct.pack('<I%sB'%length, length, *_x))
      else:
        buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.lang.language
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      if python3:
        buff.write(struct.pack('<I%sB'%length, length, *_x))
      else:
        buff.write(struct.pack('<I%ss'%length, length, _x))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      if self.activation is None:
        self.activation = pal_interaction_msgs.msg.ASRActivation()
      if self.model is None:
        self.model = pal_interaction_msgs.msg.ASRLangModelMngmt()
      if self.lang is None:
        self.lang = pal_interaction_msgs.msg.ASRLanguage()
      end = 0
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sb'%length
      start = end
      end += struct.calcsize(pattern)
      self.requests = numpy.frombuffer(str[start:end], dtype=numpy.int8, count=length)
      _x = self
      start = end
      end += 2
      (_x.activation.action, _x.model.action,) = _struct_2b.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.model.modelName = str[start:end].decode('utf-8')
      else:
        self.model.modelName = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.lang.language = str[start:end].decode('utf-8')
      else:
        self.lang.language = str[start:end]
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
_struct_2b = struct.Struct("<2b")
