import { Module } from '@nestjs/common'
import { BindingController } from './binding.controller'

@Module({
  controllers: [BindingController],
})
export class BindingModule {}
