
import { Provider, observer, inject } from 'mobx-react'
import styled from 'styled-components'
import isemail from 'isemail'
import React from 'react'

import { Step } from '../../constant'
import store from '../store'

const Header = styled.header`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 15vh;
  color: gray;
`

const Main = styled.main`
  padding: 2vh 3vw;
`

const Input = styled.input`
  width: 100%;
  height: 5vh;
  margin-top: 1vh;
  padding: 3px;
  border: 1px solid gray;
`

const Button = styled.button`
  width: 100%;
  height: 6vh;
  margin-top: 3vh;
  border: 1px solid gray;
`

const Footer = styled.footer`
  margin-top: 3vh;
  color: red;
`

@observer
class Form extends React.Component<any, any> {
  onClickSend = async () => {
    if (this.validate()) {
      await store.send()
    }
  }

  validate (): boolean {
    const {
      name,
      email,
      email2
    } = store

    if (!(name && name.length >= 3)) {
      store.setErrMsg('name is required and at least 3 chars !')
      return false
    }

    if (!email) {
      store.setErrMsg('email is required !')
      return false
    }

    if (!email2) {
      store.setErrMsg('please confirm email !')
      return false
    }

    if (!isemail.validate(email)) {
      store.setErrMsg('please type the correct email !')
      return false
    }

    if (email !== email2) {
      store.setErrMsg('please confirm your email !')
      return false
    }

    store.setErrMsg('')
    return true
  }

  get disabled () {
    return store.step === Step.Sending
  }

  get btnText () {
    switch (store.step) {
      case Step.Sending:
        return 'Sending, please wait.'
      default:
        return 'Send'
    }
  }

  get footer () {
    if (store.errMsg) {
      return (
        <Footer>
          {store.errMsg}
        </Footer>
      )
    }

    return null
  }

  setName (e: any) {
    store.name = e.target.value
  }

  setEmail (e: any) {
    store.email = e.target.value.trim()
  }

  setEmail2 (e: any) {
    store.email2 = e.target.value.trim()
  }

  render () {
    return (
      <>
        <Header>
          Request an invite
        </Header>
        <Main>
          <Input
            placeholder='full name'
            value={store.name}
            onChange={this.setName}
          />
          <Input
            type='email'
            placeholder='email'
            value={store.email}
            onChange={this.setEmail}
          />
          <Input
            type='email'
            placeholder='confirm email'
            value={store.email2}
            onChange={this.setEmail2}
          />
          <Button
            disabled={this.disabled}
            onClick={this.onClickSend}
          >
            {this.btnText}
          </Button>
        </Main>
        {this.footer}
      </>
    )
  }
}

export default Form
