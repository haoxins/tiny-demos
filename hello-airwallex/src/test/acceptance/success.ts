
import puppeteer from 'puppeteer'

import { host } from '../config'

describe('# basic flow', () => {
  let browser
  let page

  beforeAll(async () => {
    let width = 375
    let height = 812

    browser = await puppeteer.launch({
      headless: false,
      slowMo: 80,
      args: [`--window-size=${width},${height}`]
    })

    page = await browser.newPage()

    await page.setViewport({ width, height })
  })

  it('success', async () => {
    await page.goto(host)

    await page.tap('#app > main > button')
    await page.waitForSelector('.ReactModal__Content > main > input')

    await page.type('.ReactModal__Content > main > input:nth-child(1)', 'haoxin')
    await page.type('.ReactModal__Content > main > input:nth-child(2)', 'hx@qq.com')
    await page.type('.ReactModal__Content > main > input:nth-child(3)', 'hx@qq.com')

    await page.click('.ReactModal__Content > main > button')
    await page.waitForSelector('.ReactModal__Content > main > h4')

    const f = () => document
      .querySelector('.ReactModal__Content > main > button')
      .textContent
    const text = await page.evaluate(f)

    await page.click('.ReactModal__Content > main > button')

    expect(text).toBe('OK')
  }, 20000)


  it('400 error', async () => {
    await page.goto(host)

    await page.tap('#app > main > button')
    await page.waitForSelector('.ReactModal__Content > main > input')

    await page.type('.ReactModal__Content > main > input:nth-child(1)', 'haoxin')
    await page.type('.ReactModal__Content > main > input:nth-child(2)', 'usedemail@airwallex.com')
    await page.type('.ReactModal__Content > main > input:nth-child(3)', 'usedemail@airwallex.com')

    await page.click('.ReactModal__Content > main > button')
    await page.waitForSelector('.ReactModal__Content > footer')

    const f = () => document
      .querySelector('.ReactModal__Content > footer')
      .textContent
    const text = await page.evaluate(f)

    expect(text).toBe('Bad Request: Email is already in use')
  }, 30000)

  afterAll(() => {
    browser.close()
  })
})
