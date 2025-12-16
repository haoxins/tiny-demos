package snippet

import (
	"encoding/json"
	"math"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
)

func TestMarshal(t *testing.T) {
	type Transaction struct {
		Amount float64
		From   string
		To     string
	}

	t1 := Transaction{
		Amount: 100,
		From:   "Alice",
		To:     "Bob",
	}

	bytes, err := json.Marshal(t1)
	assert.Nil(t, err)
	assert.Equal(t, `{"Amount":100,"From":"Alice","To":"Bob"}`, string(bytes))

	type Result struct {
		Transactions []*Transaction `json:"transactions"`
	}

	var r0 Result

	bytes, err = json.Marshal(r0)
	assert.Nil(t, err)
	assert.Equal(t, `{"transactions":null}`, string(bytes))

	r1 := Result{
		Transactions: []*Transaction{},
	}

	bytes, err = json.Marshal(r1)
	assert.Nil(t, err)
	assert.Equal(t, `{"transactions":[]}`, string(bytes))

	type Result2 struct {
		Transactions []*Transaction `json:"transactions,omitempty"`
	}

	r2 := Result2{
		Transactions: []*Transaction{},
	}

	bytes, err = json.Marshal(r2)
	assert.Nil(t, err)
	assert.Equal(t, `{}`, string(bytes))

	var r3 Result2

	bytes, err = json.Marshal(r3)
	assert.Nil(t, err)
	assert.Equal(t, `{}`, string(bytes))
}

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
	assert.Len(t, a1.Badges, 0)

	s2 := `{"badges": null}`
	a2 := Account{}
	json.Unmarshal([]byte(s2), &a2)
	assert.True(t, lo.IsNil(a2.Badges))
	assert.Len(t, a2.Badges, 0)

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
	_, e := json.Marshal(Transaction{
		Amount: math.NaN(),
	})
	assert.Equal(t, "json: unsupported value: NaN", e.Error())
	// Inf
	_, e = json.Marshal(Transaction{
		Amount: math.Inf(0),
	})
	assert.Equal(t, "json: unsupported value: +Inf", e.Error())

	var list1 []Account
	text := `[{}]`
	json.Unmarshal([]byte(text), &list1)
	assert.Len(t, list1, 1)
	assert.Len(t, list1[0].Badges, 0)
	assert.True(t, lo.IsNil(list1[0].Balance))
}

func TestOmitEmpty(t *testing.T) {
	type A struct {
		Name   string   `json:"name,omitempty"`
		Age    int      `json:"age,omitempty"`
		Valid  bool     `json:"valid,omitempty"`
		Score  float32  `json:"score,omitempty"`
		Badges []string `json:"badges,omitempty"`
	}
	a, _ := json.Marshal(A{})
	assert.Equal(t, `{}`, string(a))

	type B struct {
		Badges []string `json:"badges,omitempty"`
	}

	b, _ := json.Marshal(B{
		Badges: []string{},
	})
	assert.Equal(t, `{}`, string(b))

	type C struct {
		CreatedAt time.Time `json:"created_at"` // omitempty is useless for time.Time
	}

	c, _ := json.Marshal(C{})
	assert.Equal(t, `{"created_at":"0001-01-01T00:00:00Z"}`, string(c))
}

func TestOmitZero(t *testing.T) {
	type A struct {
		Name   string   `json:"name,omitzero"`
		Age    int      `json:"age,omitzero"`
		Valid  bool     `json:"valid,omitzero"`
		Score  float32  `json:"score,omitzero"`
		Badges []string `json:"badges,omitzero"`
	}
	a, _ := json.Marshal(A{})
	assert.Equal(t, `{}`, string(a))

	type B struct {
		Badges []string `json:"badges,omitzero"`
	}

	b, _ := json.Marshal(B{
		Badges: []string{},
	})
	assert.Equal(t, `{"badges":[]}`, string(b))

	type C struct {
		CreatedAt time.Time `json:"created_at,omitzero"`
	}

	c, _ := json.Marshal(C{})
	assert.Equal(t, `{}`, string(c))
}
