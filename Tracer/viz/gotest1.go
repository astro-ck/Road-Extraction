package main

import (
	"fmt"
	"io/ioutil"
	"os"

	"github.com/mitroadmaps/gomapinfer/common"
)

func GetAllFile(pathname string) []string {
	var path_list []string
	rd, _ := ioutil.ReadDir(pathname)
	for _, fi := range rd {
		if fi.IsDir() {
			fmt.Printf("[%s]\n", pathname+"\\"+fi.Name())
			GetAllFile(pathname + fi.Name() + "\\")
		} else {
			fmt.Println(fi.Name())
			path_list = append(path_list, pathname+"\\"+fi.Name())
		}
	}
	return path_list
}

func viz_graph(graph_dir string) {

	var rect common.Rectangle
	if os.Args[1] == "chicago" {
		rect = common.Rectangle{
			common.Point{-1024, -2048},
			common.Point{0, -1024},
		}
	} else if os.Args[1] == "la" || os.Args[1] == "ny" || os.Args[1] == "toronto" || os.Args[1] == "amsterdam" || os.Args[1] == "denver" || os.Args[1] == "kansascity" || os.Args[1] == "montreal" || os.Args[1] == "paris" || os.Args[1] == "pittsburgh" || os.Args[1] == "saltlakecity" || os.Args[1] == "tokyo" || os.Args[1] == "vancouver" || os.Args[1] == "doha" || os.Args[1] == "san diego" || os.Args[1] == "denver" || os.Args[1] == "atlanta" {
		rect = common.Rectangle{
			common.Point{-4096, -4096},
			common.Point{4096, 4096},
		}
	} else if os.Args[1] == "boston" {
		rect = common.Rectangle{
			common.Point{4096, -4096},
			common.Point{12288, 4096},
		}
	} else {
		fmt.Printf("unknown type %s\n", os.Args[1])
		rect = common.Rectangle{
			common.Point{0, 0},
			common.Point{1024, 1024},
		}
	}
	boundables := []common.Boundable{common.EmbeddedImage{
		Src:   rect.Min,
		Dst:   rect.Max,
		Image: fmt.Sprintf("E:/igarss/data/cornerdetect/testsat8192/%s.png", os.Args[1]),
	}}

	// read all graph files
	path_list := GetAllFile(graph_dir)
	for _, path := range path_list {
		fmt.Printf("%s", path)
		graph, err := common.ReadGraph(path)
		if err != nil {
			panic(err)
		}

		boundables = append(boundables, common.ColoredBoundable{graph, "yellow"})
	}
	//	graph, err := common.ReadGraph("F:/program/java/pyRoadTracer/data/result/out.graph")

	outname := "out1.svg"
	if len(os.Args) >= 4 {
		outname = os.Args[3]
	}

	if err := common.CreateSVG(outname, [][]common.Boundable{boundables}, common.SVGOptions{StrokeWidth: 2.0, Zoom: 2, Bounds: rect, Unflip: true}); err != nil {
		panic(err)
	}
}

func main() {
	viz_graph("E:/igarss/result/multitracer/san diego/")
}
