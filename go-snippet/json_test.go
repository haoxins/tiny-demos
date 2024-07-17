package snippet

import (
	"encoding/json"
	"fmt"
	"math"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/stat"
)

func TestJSON(t *testing.T) {
	type Balance struct {
		Amount float64 `json:"amount"`
	}

	type Account struct {
		Badges  []string `json:"badges"`
		Balance *Balance `json:"balance"`
	}

	type Transaction struct {
		Amount float64 `json:"amount"`
	}

	s1 := `{"": []}`
	a1 := Account{}
	json.Unmarshal([]byte(s1), &a1)
	assert.True(t, lo.IsNil(a1.Badges))
	assert.Equal(t, 0, len(a1.Badges))

	s2 := `{"badges": null}`
	a2 := Account{}
	json.Unmarshal([]byte(s2), &a2)
	assert.True(t, lo.IsNil(a2.Badges))
	assert.Equal(t, 0, len(a2.Badges))

	a1 = Account{
		Badges: nil,
	}
	b1, _ := json.Marshal(a1)
	assert.Equal(t, `{"badges":null,"balance":null}`, string(b1))

	a2 = Account{
		Badges: []string{},
	}
	b2, _ := json.Marshal(a2)
	assert.Equal(t, `{"badges":[],"balance":null}`, string(b2))

	// NaN
	v := stat.Variance([]float64{0.0}, nil)
	assert.Equal(t, true, math.IsNaN(v))
	v = stat.Variance([]float64{1.1}, nil)
	assert.Equal(t, true, math.IsNaN(v))
	fmt.Println("variance:", v)
	_, e := json.Marshal(Transaction{
		Amount: v,
	})
	assert.Equal(t, "json: unsupported value: NaN", e.Error())

	var list1 []Account
	text := `[{}]`
	json.Unmarshal([]byte(text), &list1)
	assert.Equal(t, 1, len(list1))
	assert.Len(t, list1[0].Badges, 0)
	assert.True(t, lo.IsNil(list1[0].Balance))
}

func TestOmitEmpty(t *testing.T) {
	type A struct {
		Name  string  `json:"name,omitempty"`
		Age   int     `json:"age,omitempty"`
		Valid bool    `json:"valid,omitempty"`
		Score float32 `json:"score,omitempty"`
	}
	b, _ := json.Marshal(A{})
	assert.Equal(t, `{}`, string(b))
}
