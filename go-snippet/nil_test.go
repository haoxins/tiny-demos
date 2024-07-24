package snippet

import (
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
)

func TestNil(t *testing.T) {
	type Balance struct {
		Amount float64 `json:"amount"`
	}

	type Account struct {
		Badges  []string `json:"badges"`
		Balance *Balance `json:"balance"`
	}

	nums := []int{1, 2, 3}
	assert.Len(t, nums, 3)

	nums = []int{}
	assert.Len(t, nums, 0)
	for range nums {
		t.Error("should not run")
	}
	// nil
	nums = nil
	assert.Len(t, nums, 0)
	for range nums {
		t.Error("should not run")
	}
	// append nil
	nums = append(nums, 1, 2, 3)
	assert.Len(t, nums, 3)

	var a Account
	assert.True(t, lo.IsNil(a.Badges))
	var p *Account

	assert.PanicsWithError(t, "runtime error: invalid memory address or nil pointer dereference", func() {
		p.Badges = []string{}
	})

	var list1 []Account
	assert.Len(t, list1, 0)
	assert.True(t, lo.IsNil(list1))
	var list2 []*Account
	assert.Len(t, list2, 0)
	assert.True(t, lo.IsNil(list2))
}
