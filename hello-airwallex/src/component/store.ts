
import { observable, action } from 'mobx'

import { Step } from '../constant'
import { auth } from '../api/user'

class Store {
  @observable public name: string
  @observable public email: string
  @observable public email2: string

  @observable public step: number
  @observable public errMsg: string
  @observable public showModal: boolean

  constructor () {
    this._clear()
    this.showModal = false
    this.step = Step.Init
  }

  _clear () {
    this.errMsg = ''

    this.name = ''
    this.email = ''
    this.email2 = ''
  }

  @action
  toggleModal = () => {
    this.showModal = !this.showModal
  }

  @action
  setErrMsg = (msg: string) => {
    this.step = Step.ValidationError
    this.errMsg = msg
  }

  @action
  clear = () => {
    this._clear()
  }

  @action
  send = async () => {
    const {
      name,
      email
    } = this

    this.step = Step.Sending

    const result = await auth(name, email)

    if (result.ok) {
      this.step = Step.Success
      this.clear()
    } else {
      this.step = Step.ServerError
      this.setErrMsg(result.errorMessage)
    }
  }
}

const store = new Store()

export default store
