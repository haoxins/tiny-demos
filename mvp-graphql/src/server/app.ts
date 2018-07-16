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
    orders: [Order]
  }

  type Order {
    id: String
    price: Int
    user: User
  }

  type Query {
    getUsers(name: String): [User]
    getOrders: [Order]
  }

  type Mutation {
    addUser(name: String, age: Int): User
  }
`

const resolvers = {
  Query: {
    getUsers: (root: any, args: any, ctx: any, info: any) => {
      console.debug('args:', args)
      const name: string = args.name || ''
      return users.filter(u => u.name.includes(name))
    }
  },

  Mutation: {
    addUser: (root: any, args: any, ctx: any, info: any) => {
      users.push(args)
    }
  }
}

const server = new ApolloServer({ typeDefs, resolvers })
;(async () => {
  const { url } = await server.listen()
  console.log(`🚀 Server ready at ${url}`)
})()
