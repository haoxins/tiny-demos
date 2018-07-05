
const HtmlWebpackPlugin = require('html-webpack-plugin')
const base = require('./webpack.base')
const merge = require('webpack-merge')
const webpack = require('webpack')
const { join } = require('path')

const config = merge(base, {
  mode: 'production',
  output: {
    publicPath: 'https://example.com/project-name/',
    filename: '[name].[chunkhash:18].js'
  }
})

config.plugins = [
  ...(config.plugins || []),
  new webpack.DefinePlugin({
    'process.env': {
      NODE_ENV: '"production"'
    }
  }),
  new webpack.EnvironmentPlugin(['NODE_ENV']),
  new HtmlWebpackPlugin({
    template: join(__dirname, '../public/index.html'),
    filename: 'index.html',
    inject: true
  })
]

module.exports = config
