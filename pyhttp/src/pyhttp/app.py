from robyn import Robyn

app = Robyn(__file__)


@app.get("/")
async def h(req):
    return "Hello, Robyn!"


app.start(port=8080)
