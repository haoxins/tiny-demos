
import { auth } from '../../../api/user'

import faker from 'faker'

describe('# user APIs', () => {
  describe('## auth', () => {
    it('should ok', async () => {
      const email = faker.internet.email()
      const data = await auth('haoxin', email)
      expect(data.message).toBe('Registered')
    })

    it('should 400', async () => {
      const email = 'usedemail@airwallex.com'
      const data = await auth('haoxin', email)
      expect(data.errorMessage).toBe('Bad Request: Email is already in use')
    })
  })
})
