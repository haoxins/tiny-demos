import { Module } from '@nestjs/common'
import { BindingService } from './binding.service'
import { BindingController } from './binding.controller'

@Module({
  providers: [BindingService],
  controllers: [BindingController],
})
export class BindingModule {}
