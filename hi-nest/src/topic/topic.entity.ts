import { Entity, PrimaryGeneratedColumn, Column } from 'typeorm'

import { MaxLength } from 'class-validator'

export enum BindingType {
  Hero = 'hero',
  Story = 'story',
  Topic = 'topic',
}

@Entity()
export class Topic {
  @PrimaryGeneratedColumn() id: number

  @MaxLength(255, { always: true })
  @Column()
  title: string

  @Column({
    name: 'source_type',
    type: 'enum',
    enum: BindingType,
  })
  sourceType: BindingType

  @Column()
  source: number

  @Column({
    name: 'target_type',
    type: 'enum',
    enum: BindingType,
  })
  targetType: BindingType

  @Column()
  target: number
}
