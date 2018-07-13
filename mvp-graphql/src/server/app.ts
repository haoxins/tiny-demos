import { ApolloServer, gql } from 'apollo-server'

const users = [
  {
    name: 'hx',
    age: 19
  },
  {
    name: 'ms',
    age: 18
  }
]

const typeDefs = gql`
  type User {
    name: String
    age: Int
  }

  type Query {
    users: [User]
  }
`

const resolvers = {
  Query: {
    users: () => users
  }
}

const server = new ApolloServer({ typeDefs, resolvers })
;(async () => {
  const { url } = await server.listen()
  console.log(`🚀 Server ready at ${url}`)
})()
