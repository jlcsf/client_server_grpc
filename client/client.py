import grpc
import sys
import time
import io

sys.path.append("../output")

import agent_pb2_grpc as AgentRPC
import agent_pb2 as Agent
import session_pb2 as Session
import resources_pb2 as Resources
import genop_pb2 as Genop
import profiling_pb2 as Profiling
import image_pb2 as Image
import PIL.Image


class VaccelClient(object):
    """
    Client for gRPC functionality
    """
    
    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port))
        self.stub = AgentRPC.VaccelAgentStub(self.channel)


    def create_session(self, flags):
        request = Session.CreateSessionRequest(flags=flags)
        response = self.stub.CreateSession(request)
        session_id = response.session_id
        print(f'Session created with session_id: {session_id}')
        return session_id
    
    def update_session(self, session_id, flags):
        request = Session.UpdateSessionRequest(session_id=session_id, flags=flags)
        response = self.stub.UpdateSession(request)
        print(f'Session {session_id} updated with flags: {flags}')
        return response

    def destroy_session(self, session_id):
        request = Session.DestroySessionRequest(session_id=session_id)
        response = self.stub.DestroySession(request)
        print(f'Session {session_id} destroyed')
        return response

    def create_resource(self, model):
        request = Resources.CreateResourceRequest()
        if isinstance(model, Resources.CreateTensorflowModelRequest):
            request.model.tf.CopyFrom(model)
        elif isinstance(model, Resources.CreateCaffeModelRequest):
            request.model.caffe.CopyFrom(model)
        elif isinstance(model, Resources.CreateTensorflowSavedModelRequest):
            request.model.tf_saved.CopyFrom(model)
        elif isinstance(model, Resources.CreateSharedObjRequest):
            request.model.shared_obj.CopyFrom(model)
        elif isinstance(model, Resources.CreateTorchSavedModelRequest):
            request.model.torch_saved.CopyFrom(model)
        else:
            pass
        response = self.stub.CreateResource(request)
        print(f'Resources ID {response.resource_id} created')
        return response.resource_id
    
    def register_resource(self, resource_id, session_id):
        request = Resources.RegisterResourceRequest(resource_id = resource_id, session_id = session_id)
        response = self.stub.RegisterResource(request)
        print(f'Session {session_id} registered with resource ID {resource_id}')
        return response
    
    def unregister_resource(self, resource_id, session_id):
        request = Resources.UnregisterResourceRequest(resource_id = resource_id, session_id = session_id)
        response = self.stub.UnregisterResource(request)
        print(f'Session {session_id} unregistered with resource ID {resource_id}')
        return response
    
    def destroy_resource(self, resource_id):
        request = Resources.DestroyResourceRequest(resource_id=resource_id)
        response = self.stub.DestroyResource(request)
        print(f'Resources {resource_id} destroyed')
        return response
    
    def image_classification(self, session_id, image):
        request = Image.ImageClassificationRequest(session_id = session_id, image = image)
        response = self.stub.ImageClassification(request)
        return response.tags
        

if __name__ == '__main__':
    client = VaccelClient()
    print("client created")
    try:

        session_id = client.create_session(1)
        print("session created")

        response = client.update_session(session_id, 2)
        print("session updated")

        resource_id = client.create_resource(1)
        print("resource created")

        response = client.register_resource(resource_id, session_id)
        print("resource registered")

        response = client.unregister_resource(resource_id, session_id)
        print("resource unregistered")

        response = client.destroy_resource(resource_id)
        print("resource destroyed")

        
        image_path = 'example.jpg'
        img = PIL.Image.open(image_path)
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='JPEG')
        img_bytes = img_byte_array.getvalue()
        
        response = client.image_classification(session_id, img_bytes)
        print(response.decode('utf-8'))
        
        response = client.destroy_session(session_id)
        print("session destroyed")
        

    except Exception as e:
        print(f"Error: {e}")
        client.channel.close()
        print("channel closed")
    finally:
        client.channel.close()
        print("channel closed")
    
    