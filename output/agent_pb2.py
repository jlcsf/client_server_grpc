# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: agent.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import session_pb2 as session__pb2
import resources_pb2 as resources__pb2
import image_pb2 as image__pb2
import tensorflow_pb2 as tensorflow__pb2
import torch_pb2 as torch__pb2
import genop_pb2 as genop__pb2
import profiling_pb2 as profiling__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0b\x61gent.proto\x12\x06vaccel\x1a\rsession.proto\x1a\x0fresources.proto\x1a\x0bimage.proto\x1a\x10tensorflow.proto\x1a\x0btorch.proto\x1a\x0bgenop.proto\x1a\x0fprofiling.proto\"\r\n\x0bVaccelEmpty2\x89\x0b\n\x0bVaccelAgent\x12L\n\rCreateSession\x12\x1c.vaccel.CreateSessionRequest\x1a\x1d.vaccel.CreateSessionResponse\x12\x42\n\rUpdateSession\x12\x1c.vaccel.UpdateSessionRequest\x1a\x13.vaccel.VaccelEmpty\x12\x44\n\x0e\x44\x65stroySession\x12\x1d.vaccel.DestroySessionRequest\x1a\x13.vaccel.VaccelEmpty\x12O\n\x0e\x43reateResource\x12\x1d.vaccel.CreateResourceRequest\x1a\x1e.vaccel.CreateResourceResponse\x12H\n\x10RegisterResource\x12\x1f.vaccel.RegisterResourceRequest\x1a\x13.vaccel.VaccelEmpty\x12L\n\x12UnregisterResource\x12!.vaccel.UnregisterResourceRequest\x1a\x13.vaccel.VaccelEmpty\x12\x46\n\x0f\x44\x65stroyResource\x12\x1e.vaccel.DestroyResourceRequest\x1a\x13.vaccel.VaccelEmpty\x12^\n\x13ImageClassification\x12\".vaccel.ImageClassificationRequest\x1a#.vaccel.ImageClassificationResponse\x12\x46\n\x0bImageDetect\x12\x1a.vaccel.ImageDetectRequest\x1a\x1b.vaccel.ImageDetectResponse\x12I\n\x0cImageSegment\x12\x1b.vaccel.ImageSegmentRequest\x1a\x1c.vaccel.ImageSegmentResponse\x12?\n\tImagePose\x12\x18.vaccel.ImagePoseRequest\x1a\x18.vaccel.ImagePoseRequest\x12\x42\n\nImageDepth\x12\x19.vaccel.ImageDepthRequest\x1a\x19.vaccel.ImageDepthRequest\x12^\n\x13TensorflowModelLoad\x12\".vaccel.TensorflowModelLoadRequest\x1a#.vaccel.TensorflowModelLoadResponse\x12\x64\n\x15TensorflowModelUnload\x12$.vaccel.TensorflowModelUnloadRequest\x1a%.vaccel.TensorflowModelUnloadResponse\x12[\n\x12TensorflowModelRun\x12!.vaccel.TensorflowModelRunRequest\x1a\".vaccel.TensorflowModelRunResponse\x12^\n\x13TorchJitloadForward\x12\".vaccel.TorchJitloadForwardRequest\x1a#.vaccel.TorchJitloadForwardResponse\x12\x34\n\x05Genop\x12\x14.vaccel.GenopRequest\x1a\x15.vaccel.GenopResponse\x12@\n\tGetTimers\x12\x18.vaccel.ProfilingRequest\x1a\x19.vaccel.ProfilingResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'agent_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_VACCELEMPTY']._serialized_start=129
  _globals['_VACCELEMPTY']._serialized_end=142
  _globals['_VACCELAGENT']._serialized_start=145
  _globals['_VACCELAGENT']._serialized_end=1562
# @@protoc_insertion_point(module_scope)
