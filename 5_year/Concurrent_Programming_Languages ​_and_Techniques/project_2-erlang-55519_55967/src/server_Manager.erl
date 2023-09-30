%%%-------------------------------------------------------------------
%%% @author Asus
%%% @copyright (C) 2022, <COMPANY>
%%% @doc
%%%
%%% @end
%%% Created : 13. nov 2022 19:51
%%%-------------------------------------------------------------------
-module(server_Manager).

-import(lists,[append/2]).

-author("Asus").

%% API
-export([start_Manager/0, server_Manager/2, register_user/3, checkUser/2,
  new_repository/5, checkRepo/2, checkRepoAdmin/3, checkRepoUser/3, addUserToRepo/4,
  addFileToRepo/5, checkRepoFiles/3, checkFiles/2, push_files/5, updateFilesInRepo/5,
  updateFiles/3, updateFile/4,get_Status/4]).

start_Manager() ->
  io:format("server running\n"),
  Manager = spawn(?MODULE, server_Manager, [[], []]),
  Manager.

server_Manager(Repositories, Users) ->
  receive
    {register, UserName, PId} ->
      io:format("server running\n"),
      server_Manager(Repositories, register_user(UserName, Users, PId));
    {new, UserName, Repository_name, PId} ->
      io:format("server running\n"),
      server_Manager(new_repository(UserName, Repository_name, Repositories, Users, PId), Users);
    {invite, Admin, UserToBeAdded, Repository, PId} ->
      server_Manager(add_user_repository(Admin,
        UserToBeAdded,
        Repository,
        Repositories,
        Users,
        PId),
        Users);
    {add, User, Repository, FileName, PId} ->
      server_Manager(add_file_repository(User, Repository, FileName, Repositories, PId),
        Users);
    {push, User, Repository, FilesName, PId} ->
      server_Manager(push_files(User, Repository, FilesName, Repositories, PId), Users);
    {status,Username,Repository,PId} ->
      server_Manager(get_Status(Username,Repository,Repositories,PId),Users)

  end.

get_Status(Username,Repository,Repositories,PId)->
  UAlreadyRepo = checkRepoUser(Username, Repository, Repositories),
  RExists = checkRepo(Repository, Repositories),
  if
    UAlreadyRepo == false ->
      PId ! user_does_not_belong_to_the_repository,
      Repositories;
    RExists == false ->
      PId ! repository_does_not_exists,
      Repositories;
    true -> Repo = getRepo(Repository,Repositories),
      PId ! {status,Repo},
      Repositories
  end.

getRepo(Repository_name, [{N, A, U, F} | T]) ->
  if Repository_name == N ->
    {N, A, U, F};
    true ->
      checkRepo(Repository_name, T)
  end.


push_files(User, Repository, FilesName, Repositories, PId) ->
  UAlreadyRepo = checkRepoUser(User, Repository, Repositories),
  FAlreadyRepo = checkRepoFiles(FilesName, Repository, Repositories),
  RExists = checkRepo(Repository, Repositories),
  if RExists == false ->
    PId ! repository_does_not_exists,
    Repositories;
    FAlreadyRepo == false ->
      PId ! file_or_files_are_not_in_this_repository,
      Repositories;
    UAlreadyRepo == false ->
      PId ! user_does_not_belong_to_the_repository,
      Repositories;
    true ->
      R = updateFilesInRepo(User, FilesName, Repository, [], Repositories),
      PId ! true,
      R
  end.

updateFilesInRepo(User, FilesName, Repository, Acc, [{N, A, U, F} | T]) ->
  if Repository == N ->
    UpdatedFiles = updateFiles(User, FilesName, F),
    NewR = {N, A, U, UpdatedFiles},
    lists:append(Acc, lists:append([NewR],T));
    true ->
      updateFilesInRepo(User, FilesName, Repository, lists:append(Acc, [{N,A,U,F}]), T)
  end.
updateFiles(_,[],Files) ->
  Files;
updateFiles(User, [H | T], Files) ->
  UpdatedFile = updateFile(User, H, Files, []),
  updateFiles(User, T, UpdatedFile).

updateFile(User, File, [{N, U, V} | T], Acc) ->
  if File == N ->
    NewFile = {N, User, V + 1},
    lists:append(Acc, lists:append([NewFile], T));
    true ->
      updateFile(User, File, T, lists:append(Acc, [{N,U,V}]))
  end.

checkRepoFiles(_, _, []) ->
  false;
checkRepoFiles(FilesName, Repository, [{N, _, _, F} | T]) ->
  if N == Repository ->
    Check = checkFiles(FilesName, F),
    Check;
    true ->
      checkRepoFiles(FilesName, Repository, T)
  end.

checkFiles([], _) ->
  true;
checkFiles([H | T], Files) ->
  Check = checkFile(H, Files),
  if Check == true ->
    checkFiles(T, Files);
    true ->
      false
  end.

checkFile(_, []) ->
  false;
checkFile(FileName, [{N,_,_} | T]) ->
  if FileName == N ->
    true;
    true ->
      checkFile(FileName, T)
  end.

add_file_repository(User, Repository, FileName, Repositories, PId) ->
  FAlreadyRepo = checkRepoFiles([FileName], Repository, Repositories),
  UAlreadyRepo = checkRepoUser(User, Repository, Repositories),
  RExists = checkRepo(Repository, Repositories),
  if RExists == false ->
    PId ! repository_does_not_exists,
    Repositories;
    UAlreadyRepo == false ->
      PId ! user_does_not_belong_to_the_repository,
      Repositories;
    FAlreadyRepo == true ->
      PId ! file_already_exists,
      Repositories;
    true ->
      R = addFileToRepo(User, FileName, Repository, [], Repositories),
      PId ! true,
      R
  end.

addFileToRepo(User, File, Repository, Acc, [{N, A, U, F} | T]) ->
  if Repository == N ->
    NFile = lists:append(F, [{File, User, 1}]),
    NewR = {N, A, U, NFile},
    lists:append(Acc, lists:append([NewR],T));
    true ->
      addUserToRepo(User, Repository, lists:append(Acc, [{N,A,U,F}]), T)
  end.

add_user_repository(Admin, UserToBeAdded, Repository, Repositories, Users, PId) ->
  RAExists = checkRepoAdmin(Admin, Repository, Repositories),
  UAlreadyRepo = checkRepoUser(UserToBeAdded, Repository, Repositories),
  RExists = checkRepo(Repository, Repositories),
  UExists = checkUser(UserToBeAdded, Users),

  if RExists == false ->
    PId ! repository_does_not_exists,
    Repositories;
    UExists == false ->
      PId ! user_does_not_exists,
      Repositories;
    RAExists == false ->
      PId ! not_the_admin,
      Repositories;
    UAlreadyRepo == true ->
      PId ! user_already_added_to_repository,
      Repositories;
    true ->
      R = addUserToRepo(UserToBeAdded, Repository, [], Repositories),
      PId ! true,
      R
  end.

addUserToRepo(User, Repository, Acc, [{N, A, U, F} | T]) ->
  if Repository == N ->
    NewU = lists:append(U, [User]),
    NewR = {N, A, NewU, F},
    lists:append(Acc, lists:append([NewR],T));
    true ->
      addUserToRepo(User, Repository, lists:append(Acc, [{N, A, U, F}]), T)
  end.
checkRepoUser(_, _, []) ->
  false;
checkRepoUser(User, Repository_name, [{N, _, U, _} | T]) ->
  if Repository_name == N ->
    Exists = checkUser(User, U),
    Exists;
    true ->
      checkRepoUser(User, Repository_name, T)
  end.

checkRepoAdmin(_, _, []) ->
  false;
checkRepoAdmin(Admin, Repository_name, [{N, A, _, _} | T]) ->
  if (Repository_name == N) and (Admin == A) ->
    true;
    true ->
      checkRepoAdmin(Admin, Repository_name, T)
  end.

new_repository(UserName, Repository_name, Repositories, Users, PId) ->
  RExists = checkRepo(Repository_name, Repositories),
  UExists = checkUser(UserName, Users),
  if RExists == true ->
    PId ! repository_already_exists,
    Repositories;
    UExists == false ->
      PId ! user_does_not_exists,
      Repositories;
    true ->
      R = lists:append(Repositories, [{Repository_name, UserName, [UserName], []}]),
      PId ! true,
      R
  end.

checkRepo(_, []) ->
  false;
checkRepo(Repository_name, [{N, _, _, _} | T]) ->
  if Repository_name == N ->
    true;
    true ->
      checkRepo(Repository_name, T)
  end.

register_user(UserName, Users, PId) ->
  Exists = checkUser(UserName, Users),
  if Exists == true ->
    PId ! false,
    Users;
    true ->
      U = lists:append(Users, [UserName]),
      PId ! true,
      U
  end.

checkUser(_, []) ->
  false;
checkUser(UserName, [H | T]) ->
  if UserName == H ->
    true;
    true ->
      checkUser(UserName, T)
  end.