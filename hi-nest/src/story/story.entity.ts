
import { MaxLength } from 'class-validator'
@Entity()
export class Story {
  @PrimaryGeneratedColumn() id: number

  @MaxLength(255, { always: true })
  @Column()
  title: string
}
