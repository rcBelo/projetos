package builder

import (
	"cpl_go_proj22/graph"
	"cpl_go_proj22/parser"
	"cpl_go_proj22/utils"
	"fmt"
	"strings"
	"time"
)

type MsgType = int

const (
	BuildSuccess MsgType = iota
	BuildError
)

type Msg struct {
	Type MsgType
	//TODO: May add more here fields here.
}

func nodeLeaf(name string, leafch chan bool, channels map[string]chan bool, me graph.Vertex) {
	fmt.Println("leaf", name, "lauched")
	<-leafch
	_, er := utils.Status(name)
	if er != nil {
		fmt.Println("vou compilar", name)
		_, er = utils.Build(name)
	}

	for _, i := range me.Predecessor {
		fmt.Println(name, "vou dizer q", i.Key, "pode compilar")
		channels[i.Key] <- true

	}

}

func node(name string, channels map[string]chan bool, me graph.Vertex) {
	fmt.Println("node", name, "lauched")
	for len(channels[name]) != cap(channels[name]) {
	}
	newTime, _ := utils.Build(name)
	fmt.Println("node", name, "compile at", newTime)
	for _, i := range me.Predecessor {
		fmt.Println(name, "vou dizer q", i.Key, "pode compilar")
		channels[i.Key] <- true
	}

	for len(channels[name]) > 0 {
		<-channels[name]
	}
	for {
		ctime := newTime
		select {
		case <-channels[name]:
			ctime, _ := utils.Build(name)
			fmt.Println("node", name, "recompile at", ctime)
			for _, i := range me.Predecessor {
				fmt.Println(name, "vou dizer q", i.Key, "pode recompilar")
				channels[i.Key] <- true
			}
			newTime = ctime
		default:
			fTime, er := utils.Status(name)
			if er != nil || fTime != ctime {
				fmt.Println(ctime, fTime)
				ctime, _ := utils.Build(name)
				fmt.Println("node", name, "recompile at", ctime)
				for _, i := range me.Predecessor {
					fmt.Println(name, "vou dizer q", i.Key, "pode recompilar")
					channels[i.Key] <- true
				}
				newTime = ctime
			}
		}
	}
}

func nodeRoot(name string, rootch chan *Msg, channels map[string]chan bool, me graph.Vertex) {
	fmt.Println("root", name, "lauched")
	for len(channels[name]) != cap(channels[name]) {
		//fmt.Println(name, "nr files compile", len(channels[name]), "waint to be", cap(channels[name]))
	}

	newTime, _ := utils.Build(name)
	fmt.Println("root compile at", newTime)
	rootch <- &Msg{BuildSuccess}

	for len(channels[name]) > 0 {
		<-channels[name]
	}
	for {
		ctime := newTime
		select {
		case <-channels[name]:
			ctime, _ := utils.Build(name)
			fmt.Println("root recompile at", ctime)
			rootch <- &Msg{BuildSuccess}
			newTime = ctime
		default:
			fTime, er := utils.Status(name)
			if er != nil || fTime != ctime {
				//fmt.Println(er)
				fmt.Println(ctime, fTime)
				ctime, _ := utils.Build(name)
				fmt.Println("root recompile at", ctime)
				rootch <- &Msg{BuildSuccess}
				newTime = ctime
			}
		}
	}
}

func MakeController(file *parser.DepFile) chan *Msg {
	reqCh := make(chan *Msg)
	rootCh := make(chan *Msg)

	//fmt.Println(file)
	fmt.Println("comecou")
	dependencies := &graph.Graph{}
	for _, v := range file.Rules {
		first := true
		var from string
		for _, d := range strings.Split(v.String(), " ") {
			if d != "<-" && !first {
				dependencies.AddVertex(d)
				dependencies.AddEdge(from, d)
			}
			if first {
				from = d
				first = false
				dependencies.AddVertex(from)
			}
		}
	}
	dependencies.Print()
	channels := map[string]chan bool{}
	topologicOrder, nrLeafs := dependencies.TopologicOrder()
	for i, t := range topologicOrder {
		fmt.Println(i, "     ", t)
		channels[t] = make(chan bool, len(dependencies.GetAdjacent(t)))
	}

	leafCh := make(chan bool, nrLeafs)
	for i := 0; i < nrLeafs; i++ {
		go nodeLeaf(topologicOrder[i], leafCh, channels, *dependencies.GetVertex(topologicOrder[i]))
	}
	for i := nrLeafs; i < len(topologicOrder)-1; i++ {
		go node(topologicOrder[i], channels, *dependencies.GetVertex(topologicOrder[i]))
	}
	go nodeRoot(topologicOrder[len(topologicOrder)-1], rootCh, channels, *dependencies.GetVertex(topologicOrder[len(topologicOrder)-1]))

	time.Sleep(100 * time.Millisecond)

	//TODO: You may want to change this type
	// TODO: Startup system that emits outcome of build on rootCh, triggered by an output on leafCh
	go func() {
		for i := 0; i < nrLeafs; i++ {
			leafCh <- true
		}
		for {
			<-reqCh
			m := <-rootCh
			switch m.Type {
			case BuildSuccess:
				reqCh <- m
				break
			case BuildError:
				reqCh <- m
				break
			}
		}

	}()
	return reqCh
}
