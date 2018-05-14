
import { request } from './base'

export async function auth (name: string, email: string) {
  const body = await request
    .post('/prod/fake-auth')
    .send({
      name,
      email
    })
    .json(false) // notice: response is not strict json

  if (typeof body === 'string') {
    return {
      message: body,
      ok: true
    }
  }

  return body
}
