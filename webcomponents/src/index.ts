import '@webcomponents/webcomponentsjs'

class MyClient {
  constructor(token: string) {
    if (!token) {
      throw new TypeError('token required.')
    }
  }
}

export default MyClient
