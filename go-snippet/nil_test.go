package snippet

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNil(t *testing.T) {
	nums := []int{1, 2, 3}
	assert.Equal(t, 3, len(nums))
	nums = nil
	assert.Equal(t, 0, len(nums))
	nums = []int{}
	assert.Equal(t, 0, len(nums))

	var a Account
	assert.Equal(t, []string([]string(nil)), a.Badges)
	var p *Account

	assert.PanicsWithError(t, "runtime error: invalid memory address or nil pointer dereference", func() {
		p.Badges = []string{}
	})

	var list1 []Account
	assert.Equal(t, 0, len(list1))
	assert.Equal(t, []Account([]Account(nil)), list1)
	var list2 []*Account
	assert.Equal(t, 0, len(list2))
	assert.Equal(t, []*Account([]*Account(nil)), list2)
}
