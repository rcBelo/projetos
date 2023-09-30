%%%-------------------------------------------------------------------
%%% @author Yang &Ruben
%%% @copyright (C) 2022, <COMPANY>
%%% @doc
%%%
%%% @end
%%% Created : 13. 11月 2022 16:23
%%%-------------------------------------------------------------------
-module(main).
-author("Yang&Ruben").

%% API
-import(server, [start_server/0]).
-import(client, [start_good_client/2, start_bad_client/2]).
-export([start/2]).

start(NGoodClient,NBadClient)->
  Server = server:start_server(),
  create_clients(NGoodClient,NBadClient,Server).

create_clients(NGoodClient,NBadClient,Server) ->
  create_good_client(NGoodClient,Server),
  create_bad_client(NBadClient,Server).

create_good_client(NGoodClients, Server)->
  if NGoodClients > 0 ->
    client:start_good_client(Server, NGoodClients),
    create_good_client(NGoodClients-1,Server);
  true -> ok
  end.

create_bad_client(NBadClients, Server)->
  if NBadClients > 0 ->
    client:start_bad_client(Server, NBadClients),
    create_bad_client(NBadClients-1, Server);
    true -> ok
  end.
