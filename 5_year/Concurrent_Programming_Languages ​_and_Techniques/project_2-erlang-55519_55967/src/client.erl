 %%%-------------------------------------------------------------------
%%% @author Yang
%%% @copyright (C) 2022, <COMPANY>
%%% @doc
%%%
%%% @end
%%% Created : 10. 11月 2022 14:35
%%%-------------------------------------------------------------------
-module(client).
-author("Yang&Ruben").

%% API
-export([start_good_client/2,start_bad_client/2,client_loop_prep/3,client_loop/3,good_client_request/4,bad_client_request/4,extract_status/2]).

 start_good_client(ServerPid, NGoodClients)->
  io:format("GoodClient "++integer_to_list(NGoodClients)++ " created \n"),
  spawn(?MODULE, client_loop_prep, [ServerPid, true, NGoodClients]).

 start_bad_client(ServerPid, NBadClients) ->
  io:format("BadClient " ++ integer_to_list(NBadClients)++ " created \n"),
  spawn(?MODULE, client_loop_prep, [ServerPid, false, NBadClients]).

 client_loop_prep(ServerPid, IsGood, Index) ->
  if IsGood == true ->
   Username =  "GoodClient" ++ integer_to_list(Index),
   client_loop(Username, ServerPid, good_client_request(Username, "Java_Rep"++ integer_to_list(Index), self(),Index));
   true ->
    Username =  "BadClient"++ integer_to_list(Index),
    client_loop(Username, ServerPid, bad_client_request(Username, "#Java_Rep"++ integer_to_list(Index), self(),Index))
  end.

 client_loop(Username, _, []) -> io:format("~s is done\n", [Username]);
 client_loop(Username, ServerPid, [H|T]) ->
  ServerPid ! H,
  io:format("~s sent: ~p\n",[Username, H]),
  receive
   user_ok -> io:format("~s received: user registered\n", [Username]);
   already_registered -> io:format("~s received: already_registered\n", [Username]);
   repository_ok -> io:format("~s received: repository created\n",[Username]);
   repository_already_exists -> io:format("~s received: repository already exists\n",[Username]);
   user_does_not_exists -> io:format("~s received: user doesn't exist\n",[Username]);
   invite_success -> io:format("~s received: invite success\n",[Username]);
   repository_does_not_exists -> io:format("~s received: repository doesn't exist\n",[Username]);
   not_the_admin -> io:format("~s received: user is not the admin\n",[Username]);
   user_already_added_to_repository -> io:format("~s received: user already added to repository\n",[Username]);
   {add_file_success,FileName} -> io:format("~s received: file ~s added\n",[Username,FileName]);
   user_does_not_belong_to_the_repository -> io:format("~s received: that user doesn't belong to the repository\n",[Username]);
   file_already_exists -> io:format("~s received: File already exist\n",[Username]);
   {push_success,FilesName} -> io:format("~s received: ~s pushed\n",[Username,FilesName]);
   file_or_files_are_not_in_this_repository -> io:format("~s received: Some files are not in this repository\n",[Username]);
   {status,Repo} -> extract_status(Repo,Username);
    fail -> io:format("~s received: Fail",[Username])
  end,
  client_loop(Username, ServerPid, T).

 extract_status(Repositories,Username) ->
  io:format("\n ~s Conected:\n~p\n", [Username,Repositories]).

 good_client_request(Name, RepoName, Pid,Index) ->
  [
   {register, Name, Pid},
   {register,"GoodFakeUser"++ integer_to_list(Index) , Pid},
   {new,Name,RepoName, Pid},
   {invite,Name,"GoodFakeUser"++ integer_to_list(Index) , RepoName , Pid},
   {add,Name,RepoName,"File_1",Pid},
   {status,Name,RepoName,Pid},
   {add,Name,RepoName,"File_2",Pid},
   {status,Name,RepoName,Pid},
   {add,Name,RepoName,"File_3",Pid},
   {push,"GoodFakeUser"++ integer_to_list(Index),RepoName,["File_2"], Pid},
   {status,Name,RepoName,Pid}
  ].

 bad_client_request(Name, RepoName, Pid,Index) ->
  [
   {status,Name,RepoName,Pid},
   {register, Name, Pid},
   {push,"BadFakeUser"++ integer_to_list(Index),RepoName,["File_2"],Pid},
   {add,Name,RepoName,"File_2",Pid},
   {add,Name,RepoName,"File_3",Pid},
   {invite,Name,"BadFakeUser"++ integer_to_list(Index),RepoName,Pid},
   {new,Name,RepoName, Pid},
   {add,Name,RepoName,"File_1",Pid},
   {register,"BadFakeUser"++ integer_to_list(Index),Pid},
   {status,Name,RepoName,Pid}
  ].

