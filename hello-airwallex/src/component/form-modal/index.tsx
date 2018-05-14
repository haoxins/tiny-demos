
import { Provider, observer, inject } from 'mobx-react'
import ReactModal from 'react-modal'
import React from 'react'

import { Step } from '../../constant'
import store from '../store'

import Success from './success'
import Form from './form'

@observer
class FromModal extends React.Component<any, any> {
  get content () {
    if (store.step === Step.Success) {
      return <Success />
    }
    return <Form />
  }

  render () {
    const customStyles = {
      content: {
        top: '15vh',
        left: '15vw',
        right: '15vw',
        bottom: '20vh'
      }
    }

    return (
      <ReactModal
        shouldCloseOnOverlayClick={false}
        ariaHideApp={false}
        contentLabel=''
        style={customStyles}
        isOpen={store.showModal}
      >
        {this.content}
      </ReactModal>
    )
  }
}

export default FromModal
