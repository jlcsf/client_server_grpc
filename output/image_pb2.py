# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: image.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0bimage.proto\x12\x06vaccel\"?\n\x1aImageClassificationRequest\x12\x12\n\nsession_id\x18\x01 \x01(\r\x12\r\n\x05image\x18\x02 \x01(\x0c\"+\n\x1bImageClassificationResponse\x12\x0c\n\x04tags\x18\x01 \x01(\x0c\"7\n\x12ImageDetectRequest\x12\x12\n\nsession_id\x18\x01 \x01(\r\x12\r\n\x05image\x18\x02 \x01(\x0c\"#\n\x13ImageDetectResponse\x12\x0c\n\x04tags\x18\x01 \x01(\x0c\"5\n\x10ImagePoseRequest\x12\x12\n\nsession_id\x18\x01 \x01(\r\x12\r\n\x05image\x18\x02 \x01(\x0c\"!\n\x11ImagePoseResponse\x12\x0c\n\x04tags\x18\x01 \x01(\x0c\"6\n\x11ImageDepthRequest\x12\x12\n\nsession_id\x18\x01 \x01(\r\x12\r\n\x05image\x18\x02 \x01(\x0c\"\"\n\x12ImageDepthResponse\x12\x0c\n\x04tags\x18\x01 \x01(\x0c\"8\n\x13ImageSegmentRequest\x12\x12\n\nsession_id\x18\x01 \x01(\r\x12\r\n\x05image\x18\x02 \x01(\x0c\"$\n\x14ImageSegmentResponse\x12\x0c\n\x04tags\x18\x01 \x01(\x0c\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'image_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_IMAGECLASSIFICATIONREQUEST']._serialized_start=23
  _globals['_IMAGECLASSIFICATIONREQUEST']._serialized_end=86
  _globals['_IMAGECLASSIFICATIONRESPONSE']._serialized_start=88
  _globals['_IMAGECLASSIFICATIONRESPONSE']._serialized_end=131
  _globals['_IMAGEDETECTREQUEST']._serialized_start=133
  _globals['_IMAGEDETECTREQUEST']._serialized_end=188
  _globals['_IMAGEDETECTRESPONSE']._serialized_start=190
  _globals['_IMAGEDETECTRESPONSE']._serialized_end=225
  _globals['_IMAGEPOSEREQUEST']._serialized_start=227
  _globals['_IMAGEPOSEREQUEST']._serialized_end=280
  _globals['_IMAGEPOSERESPONSE']._serialized_start=282
  _globals['_IMAGEPOSERESPONSE']._serialized_end=315
  _globals['_IMAGEDEPTHREQUEST']._serialized_start=317
  _globals['_IMAGEDEPTHREQUEST']._serialized_end=371
  _globals['_IMAGEDEPTHRESPONSE']._serialized_start=373
  _globals['_IMAGEDEPTHRESPONSE']._serialized_end=407
  _globals['_IMAGESEGMENTREQUEST']._serialized_start=409
  _globals['_IMAGESEGMENTREQUEST']._serialized_end=465
  _globals['_IMAGESEGMENTRESPONSE']._serialized_start=467
  _globals['_IMAGESEGMENTRESPONSE']._serialized_end=503
# @@protoc_insertion_point(module_scope)
