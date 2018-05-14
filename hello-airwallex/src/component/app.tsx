
import DevTools from 'mobx-react-devtools'
import styled from 'styled-components'
import { onError } from 'mobx-react'
import React from 'react'

import FormModal from './form-modal'

import store from './store'

const Header = styled.header`
  display: flex;
  align-items: center;
  height: 8vh;
  padding-left: 5vw;
  border-bottom: 1px solid gray;
  color: gray;
`

const Main = styled.main`
  display: flex;
  flex-flow: column;
  justify-content: center;
  align-items: center;
  height: 80vh;
`

const Footer = styled.footer`
  display: flex;
  flex-flow: column;
  justify-content: center;
  align-items: center;
  height: 12vh;
  border-top: 1px solid gray;
  color: gray;
`

const H3 = styled.h3`
  font-size: 2em;
  color: gray;
`

const H5 = styled.h5`
  font-size: 0.75em;
  color: gray;
`

const Button = styled.button`
  margin-top: 1em;
  padding: 0.5em;
  border: 1px solid gray;
  cursor: pointer;
`

const App = () => {
  return (
    <>
      <DevTools />
      <Header>
        Broccoli & Co.
      </Header>
      <Main>
        <H3>A better way</H3>
        <H3>to enjoy every day</H3>
        <H5>Be the first to know when we launch</H5>
        <Button onClick={store.toggleModal}>
          Request an invite
        </Button>
        <FormModal />
      </Main>
      <Footer>
        <p>Made with ❤ in ShangHai </p>
        <p>©2018 Broccoli & Co. All rights reserved.</p>
      </Footer>
    </>
  )
}

onError(err => {
  console.error(err)
})

export default App
