import { Module } from '@nestjs/common'
import { StoryModule } from './story/story.module'
import { HeroModule } from './hero/hero.module'
import { TopicModule } from './topic/topic.module'
import { BindingModule } from './binding/binding.module'

@Module({
  imports: [
    HeroModule,
    StoryModule,
    TopicModule,
    BindingModule,
  ],
})
export class AppModule {}
