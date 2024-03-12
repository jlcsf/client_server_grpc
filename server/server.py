import grpc
import sys
import io

sys.path.append("../output")

from concurrent import futures
import time
import agent_pb2_grpc as pb2_grpc
import agent_pb2 as pb2
import session_pb2 as rpc_session
import resources_pb2 as rpc_resource
import image_pb2 as rpc_image
import genop_pb2 as rpc_genop
import tensorflow_pb2 as rpc_tf
import error_pb2 as rpc_error
import PIL.Image


from vaccel.session import Session
from vaccel.noop import Noop
from vaccel.image import ImageClassify, ImageDetect, ImageSegment, ImagePose, ImageDepth
import vaccel.image_genop as genimg
from vaccel.exec import Exec, Exec_with_resource
from vaccel.genop import Genop, VaccelArg, VaccelArgList, VaccelOpType
from vaccel.tensorflow import Tensor, TensorType, Node, TensorFlowModel

class VaccelService(pb2_grpc.VaccelAgentServicer):

    def __init__(self, *args, **kwargs):
        self.sess = None
        self.tf_model = None

    def CreateSession(self, request, context):
        flags = request.flags  
        print(f'Received CreateSession request with flags: {flags}')
        self.sess = Session(flags=flags)
        response = rpc_session.CreateSessionResponse(session_id=self.sess.id())
        print(f'Session created with session_id: {self.sess.id()}')
        return response

    def UpdateSession(self, request, context):
        new_flags = request.flags
        session_id = request.session_id
        self.sess.flags = new_flags
        print(f'Session has session_id: {self.sess.id()} with flag {self.sess.flags}')
        return pb2.VaccelEmpty()
    
    def DestroySession(self, request, context):
        sess_id_to_destroy = request.session_id
        if self.sess.id() == sess_id_to_destroy:
            self.sess.__del__()
        print(f'Session with ID {sess_id_to_destroy} has been destroyed')
        return pb2.VaccelEmpty()

    def CreateResource(self, request, context):
        print("creates a dummy resource id")
        response = rpc_resource.CreateResourceResponse(resource_id = 1)
        return response
    
    def RegisterResource(self, request, context):
        print("registers a dummy resource (does nothing)")
        return pb2.VaccelEmpty()

    def UnregisterResource(self, request, context):
        print("unregisters a dummy resource (does nothing)")
        return pb2.VaccelEmpty()

    def DestroyResource(self, request, context):
        print("unregisters a dummy resouce (does nothing)")
        return pb2.VaccelEmpty()

    def ImageClassification(self, request, context):
        print("This will classify an image, and then return the tag back to the client")
        image = request.image
        session_id = request.session_id
        assert session_id == self.sess.id()
        output_tag = genimg.ImageClassify.classify(image=image)
        output_tag_bytes = output_tag.encode('utf-8')
        response = rpc_image.ImageClassificationResponse(tags = output_tag_bytes)
        return response
    
    def ImageDetect(self, request, context):
        print("This will detect an image and then return the tag back to the client")
        image = request.image
        session_id = request.session_id
        output_tag = genimg.ImageDetect.detect(image=image)
        output_tag_bytes = output_tag.encode('utf-8')
        response = rpc_image.ImageDetectResponse(tags = output_tag_bytes)
        return response

    def TensorflowModelLoad(self, request, context):
        print("This loads in a tensorflow model")
        session_id = request.session_id
        model_id = request.model_id
        
        
        ### here let model id set the path of the model we want to use...
        
        self.model_tf = TensorFlowModel()
        self.model_tf.model_set_path("/usr/local/share/models/tf/lstm2")
        self.model_tf.model_register()
        self.sess.register_resource(self.model_tf)
        self.model_tf.from_session(self.sess, "/usr/local/share/models/tf/lstm2")
        
        error = 0
        if not self.model_tf.is_registered(self.sess):
            error = 1

        response = rpc_tf.TensorflowModelLoadResponse()
        
        if error: 
            response.error = rpc_error.agent_error()
            
    
        ## send back graph definition?
            
        return response
    
    def TensorflowModelUnload(self, request, context):
        print("Unload tensorflow model")
        session_id = request.session_id
        model_id = request.model_id
        
        self.sess.unregister_resource(self.model_tf)
        response = rpc_tf.TensorflowModelUnloadResponse()
        return response
    
    def TensorflowModelRun(self, request, context):
        session_id = request.session_id
        model_id = request.model_id
        run_options = request.model_id
        in_nodes = request.in_nodes
        in_tensors = request.in_tensors
        out_nodes = request.out_nodes
        
        print("request unpacked")
        

        in_nodes = [Node(node.name,node.id) for node in in_nodes]
        out_nodes = [Node(node.name, node.id) for node in out_nodes]


        ## manually create the tensor for now
        t = Tensor(in_tensors[0].dims, in_tensors[0].type)
        
        t.data = [1.0] * 30 ### manual instead of in_tensors.data
        
        t.dims = list(in_tensors[0].dims)

        in_tensors = [t]
    
        
        
        output = self.model_tf.run(self.sess, in_nodes=in_nodes, in_tensors=in_tensors, out_nodes=out_nodes)
        
        for t in output:
            print(t.__str__())
            offset = t.data
            print(offset)
        
        response = rpc_tf.TensorflowModelRunResponse()
        return response
        
    def Genop(self, request, context):
        session_id = request.session_id
        arg_read = request.read_args
        arg_write = request.write_args
        
        
        optype = int.from_bytes(arg_read[0].buf, byteorder='little')
        print("Contents of arg_read[0]:", optype)

        args_remaining = [VaccelArg(data = arg.buf) for arg in arg_read[1:]]
        
        arg_read = [VaccelArg(optype)] + args_remaining
        arg_write = [VaccelArg(data = arg.buf) for arg in arg_write]
        
        response = Genop.genop(self.sess, arg_read, arg_write)
        print("----------")
        print(arg_write[0].content)
        print("----------")
        
        arg_write_copy = []
        for i in range(len(arg_write)):
            genop_arg = rpc_genop.GenopArg()
            genop_arg.argtype = 2
            genop_arg.size = arg_write[i].size
            genop_arg.buf = bytes(arg_write[i].content, 'utf-8')
            arg_write_copy.append(genop_arg)

        response = rpc_genop.GenopResponse()
        for genop_arg in arg_write_copy:
            response.genop_result.write_args.extend([genop_arg])

        return response
    
    
    
    def Profiling(self, request, context):
        session_id = request.session_id
        response = 1
        return response
    

        
    
    # TODO: Add implemnetations for tensforflow bindings, genop, torch, profiling
                

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_VaccelAgentServicer_to_server(VaccelService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()