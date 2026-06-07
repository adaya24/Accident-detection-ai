# caller.py
from twilio.rest import Client

def make_call():
    account_sid = 'ACcc95568d1320afc76073540d6319cc74'
    auth_token = '26bd0079484574ee7cabf391c96da9c0'
    twilio_number = '+16317107815'
    destination_number = '+917499585234'

    client = Client(account_sid, auth_token)

    call = client.calls.create(
        to=destination_number,
        from_=twilio_number,
        url='http://demo.twilio.com/docs/voice.xml'
    )

    print(f"Call initiated: {call.sid}")