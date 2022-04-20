import test from 'node:test'
import assert from 'node:assert'

test('Demo test', async (t) => {
  await t.test('ok', () => {
    assert.strictEqual(1, 1)
  })
})
