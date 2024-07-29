import re
import zlib



HOST = "127.0.0.1"#"localhost"
PORT = 9898
BUFSIZE = 1024



class Message():
    """
    end: str
    roomid: int
    messageid: str
    messagecontent: list 
    message: compress bytes
    """
    def __init__(self,*args):
        self.end_list = ['encode',
                         'decode']
        self.message_id_list = ['clientroomenterrequest',
                                'clientroommasterstart',
                                'clientoutcome',
                                'clientclosesocket',
                                'clientdata',
                                'clientdataover',
                                'serverbroadcastmaster',
                                'serverbroadcaststart',
                                'serverbroadcasthint',
                                'serverbroadcastoutcome',
                                ]
        
        self.end = str(args[0])
        if self.end not in self.end_list: 
            print(f'****************************Detected illegal end!****************************')

        if self.end=='encode':
            self.room_id = int(args[1])
            self.message_id = str(args[2])
            if self.message_id not in self.message_id_list:
                print(f'****************************Detected illegal encode message id!****************************')
                print(f'****************************encode message id {self.message_id}****************************')
            self.message_content = args[3] if type(args[3])==list else []
            if type(self.message_content) != list:
                print(f'****************************Detected illegal encode message content!****************************')
                print(f'****************************encode message content {self.message_id}****************************')                
            message = '<RID:{}><MID:{}><content:{}>!'.format(self.room_id, self.message_id,self.message_content)
            self.message = comp(message)         
        elif self.end=='decode':
            self.message = args[1]
            message = dep(self.message)
            try:
                self.room_id = int(re.search(r'<RID:(\d+)>', message).group(1))  
                self.message_id = str(re.search(r'<MID:(.*?)>', message).group(1))  
                if self.message_id not in self.message_id_list:
                    print(f'****************************Detected illegal decode message id!****************************')
                    print(f'****************************decode message id {self.message_id}****************************')
                self.message_content = eval(re.search(r'<content:(.*?)>!',message).group(1))
                if type(self.message_content) != list:
                    print(f'****************************Detected illegal encode message content!****************************')
                    print(f'****************************encode message content {self.message_id}****************************')  
            except:
                print(f'****************************Detected unknown illegal decode message format!****************************')           
                print(f'****************************decode message {message}****************************')
        else:
            pass
        


def comp(message):
    return zlib.compress(message.encode('utf-8'))  
def dep(message):
    return (zlib.decompress(message)).decode('utf-8') 





    


