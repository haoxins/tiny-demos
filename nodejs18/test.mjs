import test from 'node:test'
import assert from 'assert'

test('Demo test', async (t) => {
  await t.test('ok', () => {
    assert.strictEqual(1, 1)
  })
})
