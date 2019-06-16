import { Controller, Get } from '@nestjs/common'

import { prisma } from '../generated/prisma-client'

@Controller('users')
export class UserController {
  @Get()
  async getStories(): Promise<any[]> {
    return await prisma.users()
  }
}
