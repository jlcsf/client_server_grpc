# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: resources.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0fresources.proto\x12\x06vaccel\"Q\n\x17\x43reateCaffeModelRequest\x12\x10\n\x08prototxt\x18\x02 \x01(\x0c\x12\x14\n\x0c\x62inary_model\x18\x03 \x01(\x0c\x12\x0e\n\x06labels\x18\x04 \x01(\x0c\"-\n\x1c\x43reateTensorflowModelRequest\x12\r\n\x05model\x18\x01 \x01(\x0c\"\\\n!CreateTensorflowSavedModelRequest\x12\x10\n\x08model_pb\x18\x01 \x01(\x0c\x12\x12\n\ncheckpoint\x18\x02 \x01(\x0c\x12\x11\n\tvar_index\x18\x03 \x01(\x0c\",\n\x16\x43reateSharedObjRequest\x12\x12\n\nshared_obj\x18\x01 \x01(\x0c\"-\n\x1c\x43reateTorchSavedModelRequest\x12\r\n\x05model\x18\x01 \x01(\x0c\"\xb8\x02\n\x15\x43reateResourceRequest\x12\x32\n\x02tf\x18\x01 \x01(\x0b\x32$.vaccel.CreateTensorflowModelRequestH\x00\x12\x30\n\x05\x63\x61\x66\x66\x65\x18\x02 \x01(\x0b\x32\x1f.vaccel.CreateCaffeModelRequestH\x00\x12=\n\x08tf_saved\x18\x03 \x01(\x0b\x32).vaccel.CreateTensorflowSavedModelRequestH\x00\x12\x34\n\nshared_obj\x18\x04 \x01(\x0b\x32\x1e.vaccel.CreateSharedObjRequestH\x00\x12;\n\x0btorch_saved\x18\x05 \x01(\x0b\x32$.vaccel.CreateTorchSavedModelRequestH\x00\x42\x07\n\x05model\"-\n\x16\x43reateResourceResponse\x12\x13\n\x0bresource_id\x18\x01 \x01(\x03\"B\n\x17RegisterResourceRequest\x12\x13\n\x0bresource_id\x18\x01 \x01(\x03\x12\x12\n\nsession_id\x18\x02 \x01(\r\"D\n\x19UnregisterResourceRequest\x12\x12\n\nsession_id\x18\x01 \x01(\r\x12\x13\n\x0bresource_id\x18\x02 \x01(\x03\"-\n\x16\x44\x65stroyResourceRequest\x12\x13\n\x0bresource_id\x18\x01 \x01(\x03\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'resources_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_CREATECAFFEMODELREQUEST']._serialized_start=27
  _globals['_CREATECAFFEMODELREQUEST']._serialized_end=108
  _globals['_CREATETENSORFLOWMODELREQUEST']._serialized_start=110
  _globals['_CREATETENSORFLOWMODELREQUEST']._serialized_end=155
  _globals['_CREATETENSORFLOWSAVEDMODELREQUEST']._serialized_start=157
  _globals['_CREATETENSORFLOWSAVEDMODELREQUEST']._serialized_end=249
  _globals['_CREATESHAREDOBJREQUEST']._serialized_start=251
  _globals['_CREATESHAREDOBJREQUEST']._serialized_end=295
  _globals['_CREATETORCHSAVEDMODELREQUEST']._serialized_start=297
  _globals['_CREATETORCHSAVEDMODELREQUEST']._serialized_end=342
  _globals['_CREATERESOURCEREQUEST']._serialized_start=345
  _globals['_CREATERESOURCEREQUEST']._serialized_end=657
  _globals['_CREATERESOURCERESPONSE']._serialized_start=659
  _globals['_CREATERESOURCERESPONSE']._serialized_end=704
  _globals['_REGISTERRESOURCEREQUEST']._serialized_start=706
  _globals['_REGISTERRESOURCEREQUEST']._serialized_end=772
  _globals['_UNREGISTERRESOURCEREQUEST']._serialized_start=774
  _globals['_UNREGISTERRESOURCEREQUEST']._serialized_end=842
  _globals['_DESTROYRESOURCEREQUEST']._serialized_start=844
  _globals['_DESTROYRESOURCEREQUEST']._serialized_end=889
# @@protoc_insertion_point(module_scope)
