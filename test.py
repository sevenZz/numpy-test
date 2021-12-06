from websocket import create_connection
ws = create_connection("ws://127.0.0.1:8000/ws")
ws.send('{"code":"import time\\nprint(1)\\ntime.sleep(2)\\nprint(3)"}')
#ws.send('{"code":"print(1)\\nprint(3)"}')
while True:
    result =  ws.recv()
    print("Received '%s'" % result)
