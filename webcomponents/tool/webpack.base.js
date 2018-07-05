
const { join } = require('path')

const { distPath } = require('./config')

module.exports = {
  entry: {
    index: join(__dirname, '../src/index')
  },
  output: {
    path: distPath,
    filename: '[name].js'
  },
  module: {
    rules: [{
      test: /\.tsx?$/,
      loader: 'awesome-typescript-loader',
      exclude: /node_modules/
    }, {
      enforce: 'pre',
      test: /\.js$/,
      loader: 'source-map-loader',
      exclude: /node_modules/
    }]
  },
  resolve: {
    extensions: ['.js', '.ts', '.tsx', '.json']
  },
  devtool: '#source-map'
}
