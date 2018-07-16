import ApolloClient from 'apollo-boost'
import gql from 'graphql-tag'

import 'isomorphic-fetch'

const client = new ApolloClient({
  uri: 'http://localhost:4000/'
})

;(async () => {
  const result = await client.query({
    query: gql`
      query getUsers($name: String) {
        getUsers(name: $name) {
          name
        }
      }
    `,
    variables: {
      name: 'hx'
    }
  })

  console.info((result.data as any).getUsers)

  await client.mutate({
    mutation: gql`
      mutation addUser($name: String, $age: Int) {
        addUser(name: $name, age: $age) {
          name
        }
      }
    `,
    variables: {
      name: 'hx',
      age: 19
    }
  })



})()
