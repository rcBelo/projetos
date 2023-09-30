%%%-------------------------------------------------------------------
%%% @author Asus
%%% @copyright (C) 2022, <COMPANY>
%%% @doc
%%%
%%% @end
%%% Created : 11. nov 2022 19:42
%%%-------------------------------------------------------------------
-module(server).

-author("Asus").

-import(server_Manager, [start_Manager/0]).

%% API
-export([start_server/0, server_loop/1, register_user/3, new_repository/4, invite/5,
  add/5, push/5, status/4]).

start_server() ->
  io:format("server running\n"),
  Server_manager = server_Manager:start_Manager(),
  spawn(?MODULE, server_loop, [Server_manager]).

server_loop(Server_manager) ->
  receive
    {register, UserName, Client} -> %%register
      spawn(?MODULE, register_user, [UserName, Server_manager, Client]);
    {new, UserName, Repository_name, Client} -> %% new
      spawn(?MODULE, new_repository, [UserName, Repository_name, Server_manager, Client]);
    {invite, Admin, UserToBeAdded, Repository, Client} -> %%invite
      spawn(?MODULE, invite, [Admin, UserToBeAdded, Repository, Server_manager, Client]);
    {add, UserName, Repository, FileName,Client} -> %%add
      spawn(?MODULE, add, [UserName, Repository, FileName, Server_manager, Client]);
    {push, UserName, Repository, FilesName, Client} -> %% push
      spawn(?MODULE, push, [UserName, Repository, FilesName, Server_manager, Client]);
    {status,UserName,Repository, Client} -> %%status
      spawn(?MODULE, status, [Repository, Server_manager,UserName, Client])
  end,
  server_loop(Server_manager).

register_user(UserName, Server_manager, Client) ->
  Server_manager ! {register, UserName, self()},
  receive
    true ->
      Client ! user_ok;
    false ->
      Client ! already_registered
  end.

new_repository(UserName, Repository_name, Server_manager, Client) ->
  Server_manager ! {new, UserName, Repository_name, self()},
  receive
    true ->
      Client ! repository_ok;
    repository_already_exists ->
      Client ! repository_already_exists;
    user_does_not_exists ->
      Client ! user_does_not_exists
  end.

invite(Admin, UserToBeAdded, Repository, Server_manager,Client) ->
  Server_manager ! {invite, Admin, UserToBeAdded, Repository, self()},
  receive
    true ->
      Client ! invite_success;
    repository_does_not_exists ->
      Client ! repository_does_not_exists;
    user_does_not_exists ->
      Client ! user_does_not_exists;
    not_the_admin ->
      Client ! not_the_admin;
    user_already_added_to_repository ->
      Client ! user_already_added_to_repository
  end.

add(Username, Repository, FileName, Server_manager, Client) ->
  Server_manager ! {add, Username, Repository, FileName, self()},
  receive
    true ->
      Client ! {add_file_success,FileName};
    repository_does_not_exists ->
      Client ! repository_does_not_exists;
    user_does_not_belong_to_the_repository ->
      Client ! user_does_not_belong_to_the_repository;
    file_already_exists ->
      Client ! file_already_exists
  end.

push(Username, Repository, FilesName, Server_manager, Client) ->
  Server_manager ! {push, Username, Repository, FilesName, self()},
  receive
    true ->
      Client ! {push_success,FilesName};
    user_does_not_belong_to_the_repository ->
      Client ! user_does_not_belong_to_the_repository;
    file_or_files_are_not_in_this_repository ->
      Client ! file_or_files_are_not_in_this_repository;
    repository_does_not_exists ->
      Client ! repository_does_not_exists
  end.

status(Repo ,Server_manager,Username, Client) ->
  Server_manager ! {status,Username,Repo,Client},
  receive
    {status,Repo} ->
      Client ! {status,Repo};
    _ -> Client ! fail
  end.