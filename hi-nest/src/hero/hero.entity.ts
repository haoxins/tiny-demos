import { Entity, PrimaryGeneratedColumn, Column } from 'typeorm'

import { MaxLength } from 'class-validator'
@Entity()
export class Hero {
  @PrimaryGeneratedColumn() id: number

  @MaxLength(255, { always: true })
  @Column()
  name: string
}
