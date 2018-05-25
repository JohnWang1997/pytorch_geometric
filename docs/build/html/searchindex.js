Search.setIndex({docnames:["index","modules/data","modules/datasets","modules/nn","modules/transforms","modules/utils","notes/create_dataset","notes/data_handling","notes/installation","notes/introduction"],envversion:53,filenames:["index.rst","modules/data.rst","modules/datasets.rst","modules/nn.rst","modules/transforms.rst","modules/utils.rst","notes/create_dataset.rst","notes/data_handling.rst","notes/installation.rst","notes/introduction.rst"],objects:{"torch_geometric.data":{Batch:[1,1,1,""],Data:[1,1,1,""],DataLoader:[1,1,1,""],Dataset:[1,1,1,""],InMemoryDataset:[1,1,1,""],download_url:[1,5,1,""],extract_tar:[1,5,1,""],extract_zip:[1,5,1,""]},"torch_geometric.data.Batch":{cumsum:[1,2,1,""],from_data_list:[1,3,1,""],num_graphs:[1,4,1,""]},"torch_geometric.data.Data":{apply:[1,2,1,""],cat_dim:[1,2,1,""],contains_isolated_nodes:[1,2,1,""],contains_self_loops:[1,2,1,""],contiguous:[1,2,1,""],from_dict:[1,3,1,""],is_coalesced:[1,2,1,""],is_directed:[1,2,1,""],is_undirected:[1,2,1,""],keys:[1,4,1,""],num_classes:[1,4,1,""],num_edges:[1,4,1,""],num_features:[1,4,1,""],num_nodes:[1,4,1,""],to:[1,2,1,""]},"torch_geometric.data.Dataset":{download:[1,2,1,""],get:[1,2,1,""],process:[1,2,1,""],processed_file_names:[1,4,1,""],processed_paths:[1,4,1,""],raw_file_names:[1,4,1,""],raw_paths:[1,4,1,""]},"torch_geometric.data.InMemoryDataset":{collate:[1,2,1,""],download:[1,2,1,""],get:[1,2,1,""],process:[1,2,1,""],processed_file_names:[1,4,1,""],raw_file_names:[1,4,1,""],shuffle:[1,2,1,""],split:[1,2,1,""]},"torch_geometric.datasets":{FAUST:[2,1,1,""],MNISTSuperpixels:[2,1,1,""],ModelNet:[2,1,1,""],Planetoid:[2,1,1,""],QM9:[2,1,1,""],ShapeNet:[2,1,1,""],TUDataset:[2,1,1,""]},"torch_geometric.datasets.FAUST":{download:[2,2,1,""],process:[2,2,1,""],processed_file_names:[2,4,1,""],raw_file_names:[2,4,1,""],url:[2,4,1,""]},"torch_geometric.datasets.MNISTSuperpixels":{download:[2,2,1,""],process:[2,2,1,""],processed_file_names:[2,4,1,""],raw_file_names:[2,4,1,""],url:[2,4,1,""]},"torch_geometric.datasets.ModelNet":{download:[2,2,1,""],process:[2,2,1,""],process_set:[2,2,1,""],processed_file_names:[2,4,1,""],raw_file_names:[2,4,1,""],urls:[2,4,1,""]},"torch_geometric.datasets.Planetoid":{download:[2,2,1,""],process:[2,2,1,""],processed_file_names:[2,4,1,""],raw_file_names:[2,4,1,""],url:[2,4,1,""]},"torch_geometric.datasets.QM9":{data_url:[2,4,1,""],download:[2,2,1,""],mask_url:[2,4,1,""],process:[2,2,1,""],processed_file_names:[2,4,1,""],raw_file_names:[2,4,1,""]},"torch_geometric.datasets.ShapeNet":{categories:[2,4,1,""],download:[2,2,1,""],process:[2,2,1,""],processed_file_names:[2,4,1,""],raw_file_names:[2,4,1,""],url:[2,4,1,""]},"torch_geometric.datasets.TUDataset":{download:[2,2,1,""],process:[2,2,1,""],processed_file_names:[2,4,1,""],raw_file_names:[2,4,1,""],url:[2,4,1,""]},"torch_geometric.nn":{conv:[3,0,0,"-"],pool:[3,0,0,"-"],prop:[3,0,0,"-"]},"torch_geometric.nn.conv":{ChebConv:[3,1,1,""],GATConv:[3,1,1,""],GCNConv:[3,1,1,""],NNConv:[3,1,1,""],SplineConv:[3,1,1,""]},"torch_geometric.nn.conv.ChebConv":{forward:[3,2,1,""],reset_parameters:[3,2,1,""]},"torch_geometric.nn.conv.GATConv":{forward:[3,2,1,""],reset_parameters:[3,2,1,""]},"torch_geometric.nn.conv.GCNConv":{forward:[3,2,1,""],reset_parameters:[3,2,1,""]},"torch_geometric.nn.conv.NNConv":{forward:[3,2,1,""],reset_parameters:[3,2,1,""]},"torch_geometric.nn.conv.SplineConv":{forward:[3,2,1,""],reset_parameters:[3,2,1,""]},"torch_geometric.nn.pool":{avg_pool:[3,5,1,""],avg_pool_x:[3,5,1,""],graclus:[3,5,1,""],max_pool:[3,5,1,""],max_pool_x:[3,5,1,""],voxel_grid:[3,5,1,""]},"torch_geometric.nn.prop":{AGNNProp:[3,1,1,""],GCNProp:[3,1,1,""]},"torch_geometric.nn.prop.AGNNProp":{forward:[3,2,1,""],reset_parameters:[3,2,1,""]},"torch_geometric.nn.prop.GCNProp":{forward:[3,2,1,""]},"torch_geometric.transforms":{AddSelfLoops:[4,1,1,""],Cartesian:[4,1,1,""],Center:[4,1,1,""],Compose:[4,1,1,""],FaceToEdge:[4,1,1,""],LinearTransformation:[4,1,1,""],LocalCartesian:[4,1,1,""],NNGraph:[4,1,1,""],NormalizeFeatures:[4,1,1,""],NormalizeScale:[4,1,1,""],Polar:[4,1,1,""],RadiusGraph:[4,1,1,""],RandomFlip:[4,1,1,""],RandomRotate:[4,1,1,""],RandomScale:[4,1,1,""],RandomShear:[4,1,1,""],RandomTranslate:[4,1,1,""],SamplePoints:[4,1,1,""],Spherical:[4,1,1,""],TargetIndegree:[4,1,1,""]},"torch_geometric.utils":{add_self_loops:[5,5,1,""],coalesce:[5,5,1,""],contains_isolated_nodes:[5,5,1,""],contains_self_loops:[5,5,1,""],degree:[5,5,1,""],grid:[5,5,1,""],is_coalesced:[5,5,1,""],is_undirected:[5,5,1,""],matmul:[5,5,1,""],normalized_cut:[5,5,1,""],one_hot:[5,5,1,""],remove_self_loops:[5,5,1,""],softmax:[5,5,1,""],to_undirected:[5,5,1,""]},torch_geometric:{data:[1,0,0,"-"],datasets:[2,0,0,"-"],transforms:[4,0,0,"-"],utils:[5,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","staticmethod","Python static method"],"4":["py","attribute","Python attribute"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:staticmethod","4":"py:attribute","5":"py:function"},terms:{"130k":[],"2x6":[],"3dshapenet":2,"class":[1,2,3,4],"default":[3,4,5],"float":[3,4],"function":3,"import":[4,5],"int":[3,4,5],"return":5,"static":1,"true":[1,2,3,4],For:[],The:[3,4,5],Using:[],abs:3,adapt:3,add:3,add_self_loop:5,addit:3,addselfloop:4,adj:3,aggreg:3,agnn:3,agnnprop:3,airplan:2,all:[],along:4,amazonaw:2,angl:3,appli:[1,4],approxim:[],arxiv:3,attent:3,attribut:4,automat:[],averag:3,avg_pool:3,avg_pool_x:3,axi:4,bag:2,base:3,basi:3,batch:[1,3],being:4,between:[],bia:3,bool:[3,4],callabl:[],cap:2,car:2,cartesian:4,cat:4,cat_dim:1,categori:2,center:4,chair:2,chebconv:3,chebyshev:3,chemistri:3,classfic:3,close:3,cluster:3,cnn:[],coalesc:5,coeffici:3,collat:1,com:2,compos:4,comput:[3,4,5],concat:[3,4],consist:[],contain:[],contains_isolated_nod:[1,5],contains_self_loop:[1,5],contigu:1,continu:3,contrast:4,contribut:[],convolv:3,coordin:[3,4],correspond:4,creat:0,cumsum:1,cvpr:3,cvpr_geometric_dl:2,data:[0,2,3,4],data_list:1,data_url:2,dataload:1,dataset:[0,1],deep:[0,3],deepchem:2,defin:[3,4],degre:[3,4,5],desir:5,devic:[1,5],dictionari:1,dim:3,dimens:4,dimension:[3,4],directori:[],doesn:[],dortmund:2,download:[1,2],download_url:1,dropout:3,dtype:[4,5],dure:3,each:[3,4],earphon:2,edg:[4,5],edge_attr:[1,3,4,5],edge_index:[1,3,4,5],edu:2,end:3,exactli:[],exampl:0,exist:[],expos:3,extens:0,extract_tar:1,extract_zip:1,facetoedg:4,factor:4,fals:3,fast:3,faust:2,featur:3,fei:3,figshar:2,file:2,filter:3,fix:4,flip:4,floattensor:[],folder:1,forward:3,from:[3,4,5],from_data_list:1,from_dict:1,func:1,gat:3,gatconv:3,gcnconv:3,gcnprop:3,gdb9:2,geometr:3,get:1,gilmer:3,github:2,given:[4,5],global:4,graclu:3,graph:3,graphconv:[],graphkerneldataset:2,grid:5,guitar:2,handl:0,head:3,height:5,hop:3,http:[2,3],iccv17:2,idx:1,in_channel:3,in_featur:[],index:[0,1,5],indic:5,inmemorydataset:1,input:3,instal:0,instead:4,interv:4,introduct:0,is_coalesc:[1,5],is_direct:1,is_open_splin:3,is_undirect:[1,5],item:1,its:4,kei:1,kernel:3,kernel_s:3,kilo:[],kimiyoung:2,knife:2,kwarg:1,lamp:2,laptop:2,leakyrelu:3,learn:[0,3],librari:0,lineartransform:4,link:4,list:4,local:[3,4],localcartesian:4,locat:[],log:1,longtensor:5,ls11:2,ls7:2,manifold:[],manual:[],map:[3,4],mask_url:2,master:2,matmul:5,matrix:4,max_pool:3,max_pool_x:3,maximum:4,messag:3,mixtur:[],mnist:[],mnist_superpixel:2,mnistsuperpixel:2,mode:1,model:[],modelnet10:2,modelnet40:2,modelnet:2,modul:0,molecul:[],moreov:[],morri:2,motorbik:2,mpg:2,mpi:[],mug:2,multi:3,multipl:[],must:[],name:2,ndownload:2,need:[],neg:3,negative_slop:3,neighborhood:3,network:3,neural:3,nnconv:3,nngraph:4,node:[3,4,5],none:[1,2,3,5],normal:[3,4],normalized_cut:5,normalizefeatur:4,normalizescal:4,note:0,num:4,num_class:[1,5],num_edg:1,num_featur:1,num_graph:1,num_nod:[1,5],number:[3,4,5],object:4,offlin:4,one:[],one_hot:5,open:3,oper:3,option:[3,4,5],org:3,other:4,otherwis:[],out:[],out_channel:3,out_featur:[],output:[3,5],outsid:[],over:3,own:0,packag:0,paper:3,paramet:[3,4,5],partseg:2,pass:3,path:1,peopl:2,pistol:2,planetoid:2,polar:4,pos:[1,3,4],posit:4,pre_transform:[1,2],princeton:2,print:[4,5],probabl:4,process:[1,2],process_set:2,processed_file_nam:[1,2],processed_path:1,product:3,project:2,propbabl:3,properti:[],pseudo:[3,4],qm9:2,quantum:3,radiusgraph:4,random:4,randomflip:4,randomli:4,randomrot:4,randomscal:4,randomshear:4,randomtransl:4,rang:4,raw:2,raw_file_nam:[1,2],raw_path:1,refer:0,relat:4,remove_self_loop:5,replac:4,requires_grad:3,reset_paramet:3,result:4,rocket:2,root:[1,2,3],root_weight:3,row:[],same:4,sampl:[3,4],samplepoint:4,save:4,scale:4,scatter_add_:[],semi:3,sequenc:4,sequenti:3,set:3,sever:4,shape:4,shapenet:2,shear:4,shuffl:1,singl:3,size:3,skateboard:2,slope:3,softmax:5,sourc:[1,2,3,4,5],spatial:4,specifi:[],spectral:3,spheric:4,spline:3,splinecnn:3,splineconv:3,split:1,squar:4,src:5,stanford:2,start:3,stochast:3,string:[],structur:[],sum:[],superpixel:[],supervis:3,tabl:2,take:[],tar:2,target:[4,5],targetindegre:4,tensor:[3,4,5],test:[],them:4,three:4,to_undirect:5,togeth:4,torch:[4,5],torch_geometr:0,torch_scatt:[],train:[2,3],trainabl:3,transform:[0,1,2,3],translat:4,tudataset:2,tue:2,tupl:4,two:4,type:5,uni:2,url:[1,2],use:3,used:4,using:3,util:0,valu:[4,5],vector:[],version:[],vision:2,voxel_grid:3,websit:2,weight:3,west:2,where:[3,4],whether:3,width:5,within:4,www:2,your:0,zero:[],zip:2},titles:["PyTorch Geometric documentation","torch_geometric.data","torch_geometric.datasets","torch_geometric.nn","torch_geometric.transforms","torch_geometric.utils","Create your own dataset","Data handling","Installation","Introduction by example"],titleterms:{"function":[],cites:[],content:3,convolut:3,cora:[],creat:6,data:[1,7],dataset:[2,6],document:0,exampl:9,faust:[],geometr:0,handl:7,indic:0,instal:8,introduct:9,layer:3,mnistsuperpixel:[],modul:[],own:6,pool:3,propag:3,pubm:[],pytorch:0,qm9:[],tabl:0,torch_geometr:[1,2,3,4,5],transform:4,util:5,your:6}})