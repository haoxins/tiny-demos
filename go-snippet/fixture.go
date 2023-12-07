package snippet

type Account struct {
	Badges []string `json:"badges"`
}

type Transaction struct {
	Amount float64 `json:"amount"`
}
