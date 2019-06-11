import { Module } from '@nestjs/common'
import { TypeOrmModule } from '@nestjs/typeorm'
import { AppController } from './app.controller'
import { StoryModule } from './story/story.module'
import { HeroModule } from './hero/hero.module'
import { TopicModule } from './topic/topic.module'

@Module({
  imports: [
    TypeOrmModule.forRoot({
      type: 'postgres',
      host: 'localhost',
      port: 8200,
      username: 'postgres',
      password: 'postgres',
      database: 'nest',
      entities: [__dirname + '/**/*.entity{.ts,.js}'],
      synchronize: true,
    }),
    StoryModule,
    HeroModule,
    TopicModule,
  ],
})
export class AppModule {}
