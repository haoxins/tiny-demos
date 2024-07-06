package snippet

import (
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
)

func TestNil(t *testing.T) {
	nums := []int{1, 2, 3}
	assert.Equal(t, 3, len(nums))

	nums = []int{}
	assert.Equal(t, 0, len(nums))
	for range nums {
		t.Error("should not run")
	}
	// nil
	nums = nil
	assert.Equal(t, 0, len(nums))
	for range nums {
		t.Error("should not run")
	}
	// append nil
	nums = append(nums, 1, 2, 3)
	assert.Equal(t, 3, len(nums))

	var a Account
	assert.True(t, lo.IsNil(a.Badges))
	var p *Account

	assert.PanicsWithError(t, "runtime error: invalid memory address or nil pointer dereference", func() {
		p.Badges = []string{}
	})

	var list1 []Account
	assert.Equal(t, 0, len(list1))
	assert.True(t, lo.IsNil(list1))
	var list2 []*Account
	assert.Equal(t, 0, len(list2))
	assert.True(t, lo.IsNil(list2))
}
