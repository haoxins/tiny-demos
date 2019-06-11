import { Module } from '@nestjs/common'

import { TypeOrmModule } from '@nestjs/typeorm'

import { AppController } from './app.controller'

import { HeroController } from './hero/hero.controller'
import { HeroService } from './hero/hero.service'

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
  ],
  controllers: [AppController, HeroController],
  providers: [HeroService],
})

export class AppModule {}
