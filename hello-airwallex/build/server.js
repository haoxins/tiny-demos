
process.env.NODE_ENV = 'development'

const history = require('connect-history-api-fallback')
const config = require('./webpack.dev')
const express = require('express')
const webpack = require('webpack')
const { join } = require('path')

const compiler = webpack(config)
const app = express()

app.use(history({
  index: '/dist/index.html'
}))

app.use(require('webpack-dev-middleware')(compiler, {
  publicPath: config.output.publicPath,
  noInfo: true,
  quiet: true
}))

app.use(require('webpack-hot-middleware')(compiler))

app.use(express.static(join(__dirname, '..')))

const port = 3000
app.listen(port, err => {
  if (err) {
    return console.error(err)
  }

  console.info(`Listening at localhost: ${port}`)
})
