import { Test, TestingModule } from '@nestjs/testing'
import { BindingController } from './binding.controller'

describe('Binding Controller', () => {
  let controller: BindingController

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [BindingController],
    }).compile()

    controller = module.get<BindingController>(BindingController)
  })

  it('should be defined', () => {
    expect(controller).toBeDefined()
  })
})
