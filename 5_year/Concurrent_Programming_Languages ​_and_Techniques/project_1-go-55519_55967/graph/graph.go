package graph

import (
	"fmt"
)

type Graph struct {
	vertices []*Vertex
}
type Vertex struct {
	Key         string
	Adjacent    []*Vertex
	Predecessor []*Vertex
}

func (g *Graph) GetAdjacent(k string) []*Vertex {
	return g.GetVertex(k).Adjacent
}

func (g *Graph) GetPredecessor(k string) []*Vertex {
	return g.GetVertex(k).Predecessor
}

func (g *Graph) AddVertex(k string) {
	if contains(g.vertices, k) {
		err := fmt.Errorf("vertex %v not added because it is an existing key", k)
		fmt.Println(err.Error())
	} else {
		g.vertices = append(g.vertices, &Vertex{Key: k})
	}
}

func (g *Graph) GetVertex(k string) *Vertex {
	for _, v := range g.vertices {
		if v.Key == k {
			return v
		}
	}
	return nil
}

func (g *Graph) AddEdge(from, to string) {
	fromV := g.GetVertex(from)
	toV := g.GetVertex(to)

	if fromV == nil || toV == nil {
		err := fmt.Errorf("Invalid edge (%v-->%v)", from, to)
		fmt.Println(err.Error())
	} else if contains(fromV.Adjacent, to) {
		err := fmt.Errorf("Existing dge (%v-->%v)", from, to)
		fmt.Println(err.Error())
	} else {
		fromV.Adjacent = append(fromV.Adjacent, toV)
		toV.Predecessor = append(toV.Predecessor, fromV)
	}

}

func contains(s []*Vertex, k string) bool {
	for _, v := range s {
		if k == v.Key {
			return true
		}
	}
	return false
}

func (g *Graph) Print() {
	for _, v := range g.vertices {
		fmt.Print("\n Vertex ", v.Key, ":")
		for _, v := range v.Adjacent {
			fmt.Print(" ", v.Key)
		}
	}
	fmt.Println()

	for _, v := range g.vertices {
		fmt.Print("\n Vertex ", v.Key, ":")
		for _, v := range v.Predecessor {
			fmt.Print(" ", v.Key)
		}
	}
	fmt.Println()
}

func (g *Graph) TopologicOrder() ([]string, int) {
	linearOrder := []string{}

	inDegree := map[string]int{}

	for _, v := range g.vertices {
		inDegree[v.Key] = 0
	}

	for _, v := range g.vertices {
		inDegree[v.Key] = len(v.Adjacent)
	}

	next := []string{}
	for u, v := range inDegree {
		if v != 0 {
			continue
		}
		next = append(next, u)
	}
	nrLeafs := len(next)

	for len(next) > 0 {
		u := next[0]
		next = next[1:]

		linearOrder = append(linearOrder, u)

		for _, v := range g.vertices {
			for _, a := range v.Adjacent {

				if a.Key == u {
					inDegree[v.Key]--
				}

				if inDegree[v.Key] == 0 {
					next = append(next, v.Key)
					inDegree[v.Key] = -1
				}
			}
		}
	}

	return linearOrder, nrLeafs
}
