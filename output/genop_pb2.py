# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: genop.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import error_pb2 as error__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0bgenop.proto\x12\x06vaccel\x1a\x0b\x65rror.proto\"6\n\x08GenopArg\x12\x0f\n\x07\x61rgtype\x18\x01 \x01(\r\x12\x0c\n\x04size\x18\x02 \x01(\r\x12\x0b\n\x03\x62uf\x18\x03 \x01(\x0c\"m\n\x0cGenopRequest\x12\x12\n\nsession_id\x18\x01 \x01(\r\x12#\n\tread_args\x18\x02 \x03(\x0b\x32\x10.vaccel.GenopArg\x12$\n\nwrite_args\x18\x03 \x03(\x0b\x32\x10.vaccel.GenopArg\"3\n\x0bGenopResult\x12$\n\nwrite_args\x18\x01 \x03(\x0b\x32\x10.vaccel.GenopArg\"l\n\rGenopResponse\x12$\n\x05\x65rror\x18\x01 \x01(\x0b\x32\x13.vaccel.VaccelErrorH\x00\x12+\n\x0cgenop_result\x18\x02 \x01(\x0b\x32\x13.vaccel.GenopResultH\x00\x42\x08\n\x06resultb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'genop_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_GENOPARG']._serialized_start=36
  _globals['_GENOPARG']._serialized_end=90
  _globals['_GENOPREQUEST']._serialized_start=92
  _globals['_GENOPREQUEST']._serialized_end=201
  _globals['_GENOPRESULT']._serialized_start=203
  _globals['_GENOPRESULT']._serialized_end=254
  _globals['_GENOPRESPONSE']._serialized_start=256
  _globals['_GENOPRESPONSE']._serialized_end=364
# @@protoc_insertion_point(module_scope)
