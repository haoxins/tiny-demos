
import { Provider, observer, inject } from 'mobx-react'
import styled from 'styled-components'
import React from 'react'

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
  font-size: 0.75em;
`

const Button = styled.button`
  width: 100%;
  height: 6vh;
  margin-top: 3vh;
  border: 1px solid gray;
`

const H4 = styled.h4`
  display: flex;
  justify-content: center;
  font-size: 0.75em;
  color: gray;
`

@observer
class Success extends React.Component<any, any> {
  toggleModal () {
    store.toggleModal()
  }

  render () {
    return (
      <>
        <Header>
          All done !
        </Header>
        <Main>
          <H4>You will be one of the first to experience</H4>
          <H4>Broccoli & Co. When we lanch.</H4>
          <Button
            onClick={this.toggleModal}
          >
            OK
          </Button>
        </Main>
      </>
    )
  }
}

export default Success
